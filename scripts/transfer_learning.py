import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import Config
from utils import save_pickle, load_pickle, ensure_dir

# ============================================================================
# CUSTOM DATASET
# ============================================================================

class CVEDataset(Dataset):
    """Dataset za CVE vulnerability data sa text + tabular features"""
    
    def __init__(self, texts, tabular_features, labels_clf, labels_reg, tokenizer, max_length=128):
        self.texts = texts
        self.tabular_features = tabular_features
        self.labels_clf = labels_clf
        self.labels_reg = labels_reg
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] and not pd.isna(self.texts[idx]) else ""
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tabular': torch.tensor(self.tabular_features[idx], dtype=torch.float32),
            'label_clf': torch.tensor(self.labels_clf[idx], dtype=torch.float32),
            'label_reg': torch.tensor(self.labels_reg[idx], dtype=torch.float32)
        }

# ============================================================================
# MULTI-TASK MODEL ARCHITECTURE
# ============================================================================

class CVEMultiTaskModel(nn.Module):
    """
    Multi-task learning model koji kombinuje:
    1. Fine-tuned BERT/CodeBERT za text embeddings
    2. Dense layers za tabular features
    3. Dva output head-a: klasifikacija i regresija
    """
    
    def __init__(self, model_name, n_tabular_features, dropout_rate=0.3, freeze_bert_layers=0):
        super(CVEMultiTaskModel, self).__init__()
        
        # Pretrained transformer (BERT/CodeBERT)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Opciono freezovanje početnih slojeva BERT-a
        if freeze_bert_layers > 0:
            for param in list(self.bert.parameters())[:freeze_bert_layers]:
                param.requires_grad = False
        
        bert_hidden_size = self.bert.config.hidden_size  # 768 za BERT-base
        
        # Text branch - processing BERT output
        self.text_branch = nn.Sequential(
            nn.Linear(bert_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Tabular branch - processing numerical/categorical features
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layer - combines text and tabular
        combined_size = 128 + 64  # text_branch output + tabular_branch output
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output heads
        self.classifier = nn.Linear(64, 1)  # Binary classification
        self.regressor = nn.Linear(64, 1)   # Regression
    
    def forward(self, input_ids, attention_mask, tabular):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling over sequence (alternatively could use CLS token)
        text_features = bert_output.last_hidden_state.mean(dim=1)  # [batch, 768]
        
        # Process through text branch
        text_repr = self.text_branch(text_features)  # [batch, 128]
        
        # Process tabular features
        tabular_repr = self.tabular_branch(tabular)  # [batch, 64]
        
        # Fusion
        combined = torch.cat([text_repr, tabular_repr], dim=1)  # [batch, 192]
        fused = self.fusion(combined)  # [batch, 64]
        
        # Outputs
        clf_logits = self.classifier(fused)  # [batch, 1]
        reg_output = self.regressor(fused)   # [batch, 1]
        
        return clf_logits.squeeze(), reg_output.squeeze()

# ============================================================================
# TRAINING LOOP
# ============================================================================

class MultiTaskTrainer:
    """Trainer za multi-task learning sa weighted loss"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 clf_weight=1.0, reg_weight=1.0, learning_rate=2e-5, 
                 num_epochs=5, warmup_steps=500):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.clf_weight = clf_weight
        self.reg_weight = reg_weight
        self.num_epochs = num_epochs
        
        # Loss functions
        self.clf_criterion = nn.BCEWithLogitsLoss()
        self.reg_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # History tracking
        self.history = {
            'train_loss': [], 'train_clf_loss': [], 'train_reg_loss': [],
            'val_loss': [], 'val_clf_loss': [], 'val_reg_loss': [],
            'val_accuracy': [], 'val_f1': [], 'val_r2': []
        }
    
    def train_epoch(self):
        """Jedna epoha treninga"""
        self.model.train()
        total_loss = 0
        total_clf_loss = 0
        total_reg_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            tabular = batch['tabular'].to(self.device)
            labels_clf = batch['label_clf'].to(self.device)
            labels_reg = batch['label_reg'].to(self.device)
            
            # Forward pass
            clf_logits, reg_output = self.model(input_ids, attention_mask, tabular)
            
            # Calculate losses
            clf_loss = self.clf_criterion(clf_logits, labels_clf)
            reg_loss = self.reg_criterion(reg_output, labels_reg)
            
            # Combined weighted loss
            loss = self.clf_weight * clf_loss + self.reg_weight * reg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Tracking
            total_loss += loss.item()
            total_clf_loss += clf_loss.item()
            total_reg_loss += reg_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'clf': f'{clf_loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_clf_loss = total_clf_loss / len(self.train_loader)
        avg_reg_loss = total_reg_loss / len(self.train_loader)
        
        return avg_loss, avg_clf_loss, avg_reg_loss
    
    def validate(self):
        """Validacija modela"""
        self.model.eval()
        total_loss = 0
        total_clf_loss = 0
        total_reg_loss = 0
        
        all_clf_preds = []
        all_clf_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                labels_clf = batch['label_clf'].to(self.device)
                labels_reg = batch['label_reg'].to(self.device)
                
                clf_logits, reg_output = self.model(input_ids, attention_mask, tabular)
                
                clf_loss = self.clf_criterion(clf_logits, labels_clf)
                reg_loss = self.reg_criterion(reg_output, labels_reg)
                loss = self.clf_weight * clf_loss + self.reg_weight * reg_loss
                
                total_loss += loss.item()
                total_clf_loss += clf_loss.item()
                total_reg_loss += reg_loss.item()
                
                # Predictions
                clf_preds = torch.sigmoid(clf_logits).cpu().numpy()
                all_clf_preds.extend(clf_preds)
                all_clf_labels.extend(labels_clf.cpu().numpy())
                all_reg_preds.extend(reg_output.cpu().numpy())
                all_reg_labels.extend(labels_reg.cpu().numpy())
        
        # Metrics
        clf_preds_binary = (np.array(all_clf_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_clf_labels, clf_preds_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_clf_labels, clf_preds_binary, average='binary', zero_division=0
        )
        
        r2 = r2_score(all_reg_labels, all_reg_preds)
        rmse = np.sqrt(mean_squared_error(all_reg_labels, all_reg_preds))
        
        avg_loss = total_loss / len(self.val_loader)
        avg_clf_loss = total_clf_loss / len(self.val_loader)
        avg_reg_loss = total_reg_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'clf_loss': avg_clf_loss,
            'reg_loss': avg_reg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'r2': r2,
            'rmse': rmse
        }
    
    def train(self):
        """Kompletan training loop"""
        print("\n" + "="*70)
        print("🚀 TRAINING MULTI-TASK MODEL")
        print("="*70)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            print(f"\n📍 Epoch {epoch+1}/{self.num_epochs}")
            
            # Training
            train_loss, train_clf_loss, train_reg_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_clf_loss'].append(train_clf_loss)
            self.history['train_reg_loss'].append(train_reg_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_clf_loss'].append(val_metrics['clf_loss'])
            self.history['val_reg_loss'].append(val_metrics['reg_loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_r2'].append(val_metrics['r2'])
            
            # Print metrics
            print(f"\n📊 Training   - Loss: {train_loss:.4f} | Clf: {train_clf_loss:.4f} | Reg: {train_reg_loss:.4f}")
            print(f"📊 Validation - Loss: {val_metrics['loss']:.4f} | Clf: {val_metrics['clf_loss']:.4f} | Reg: {val_metrics['reg_loss']:.4f}")
            print(f"🎯 Classification - Acc: {val_metrics['accuracy']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
            print(f"📈 Regression - R²: {val_metrics['r2']:.4f} | RMSE: {val_metrics['rmse']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = self.model.state_dict().copy()
                print(f"💾 New best model saved (val_loss: {best_val_loss:.4f})")
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        print(f"\n✅ Training completed. Best validation loss: {best_val_loss:.4f}")
        
        return self.history

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path):
    """Plotovanje training i validation metrika"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Classification metrics
    axes[0, 1].plot(history['val_accuracy'], label='Accuracy')
    axes[0, 1].plot(history['val_f1'], label='F1 Score')
    axes[0, 1].set_title('Classification Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Classification Loss
    axes[1, 0].plot(history['train_clf_loss'], label='Train Clf Loss')
    axes[1, 0].plot(history['val_clf_loss'], label='Val Clf Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Regression metrics
    axes[1, 1].plot(history['val_r2'], label='R² Score')
    axes[1, 1].set_title('Regression Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training history plot saved to {save_path}")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎯 CVE VULNERABILITY TRANSFER LEARNING")
    print("="*70)
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    df_train = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "X_train_full.csv"))
    df_val = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "X_val_full.csv"))
    
    y_train_clf = load_pickle(os.path.join(Config.OUTPUT_DIR, "y_train_clf.pkl"))
    y_train_reg = load_pickle(os.path.join(Config.OUTPUT_DIR, "y_train_reg.pkl"))
    y_val_clf = load_pickle(os.path.join(Config.OUTPUT_DIR, "y_val_clf.pkl"))
    y_val_reg = load_pickle(os.path.join(Config.OUTPUT_DIR, "y_val_reg.pkl"))
    
    # Extract features
    feature_names = load_pickle(os.path.join(Config.OUTPUT_DIR, "feature_names.pkl"))
    X_train_tabular = df_train[feature_names].values
    X_val_tabular = df_val[feature_names].values
    
    # Text data
    train_texts = df_train['description'].fillna("").astype(str).tolist() if 'description' in df_train.columns else [""] * len(df_train)
    val_texts = df_val['description'].fillna("").astype(str).tolist() if 'description' in df_val.columns else [""] * len(df_val)
    
    # Tokenizer
    print(f"\n🔤 Loading tokenizer: {Config.EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL)
    
    # Create datasets
    print("\n📦 Creating PyTorch datasets...")
    train_dataset = CVEDataset(
        train_texts, X_train_tabular, y_train_clf, y_train_reg, 
        tokenizer, max_length=Config.EMBEDDING_MAX_LENGTH
    )
    val_dataset = CVEDataset(
        val_texts, X_val_tabular, y_val_clf, y_val_reg,
        tokenizer, max_length=Config.EMBEDDING_MAX_LENGTH
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.EMBEDDING_BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EMBEDDING_BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Initialize model
    print(f"\n🏗️  Building model with {Config.EMBEDDING_MODEL}...")
    model = CVEMultiTaskModel(
        model_name=Config.EMBEDDING_MODEL,
        n_tabular_features=X_train_tabular.shape[1],
        dropout_rate=0.3,
        freeze_bert_layers=0  # Set >0 to freeze initial BERT layers
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎓 Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=Config.DEVICE,
        clf_weight=1.0,  # Weight for classification loss
        reg_weight=0.5,  # Weight for regression loss
        learning_rate=2e-5,
        num_epochs=5,
        warmup_steps=500
    )
    
    # Train
    history = trainer.train()
    
    # Save model
    model_save_path = os.path.join(Config.OUTPUT_DIR, "multitask_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'model_name': Config.EMBEDDING_MODEL,
            'n_tabular_features': X_train_tabular.shape[1],
            'dropout_rate': 0.3
        }
    }, model_save_path)
    print(f"\n💾 Model saved to {model_save_path}")
    
    # Plot training history
    plot_path = os.path.join(Config.OUTPUT_DIR, "training_history.png")
    plot_training_history(history, plot_path)
    
    print("\n✅ Transfer learning completed successfully!")

if __name__ == "__main__":
    main()