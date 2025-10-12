#!/usr/bin/env python3
"""
Cross-Species GNN Training and Testing Pipeline
Converts Jupyter notebook to standalone Python script with proper memory management
"""

import os
import sys
import pandas as pd
import torch
import gc
import traceback

# Add the src directory to path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import Biodata
import CNmodel


def safe_memory_cleanup():
    """Safely clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_results_to_excel(results, output_file="cross_species_K3d2_results.xlsx"):
    """Save results to Excel file in results folder"""
    if not results:
        print("No results to save!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns to have species info first
    columns_order = ['train_species', 'test_species', 'TP', 'FN', 'FP', 'TN', 'SN', 'SP', 'ACC', 'AUC']
    
    # Add any additional columns that might be in results
    for col in df.columns:
        if col not in columns_order:
            columns_order.append(col)
    
    df = df[columns_order]
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Full path to output file
    output_path = os.path.join(results_dir, output_file)
    
    # Save to Excel
    df.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print(f"Summary: {len(results)} test results recorded")
    return df


def train_test_pipeline():
    """Main training and testing pipeline"""
    # Define all species and their file names
    species_data = {
        'A.thaliana': {
            'fasta': 'arabidopsis_genes.fasta',
            'labels': 'arabidopsis_labels.txt'
        },
        'C.elegans': {
            'fasta': 'elegans_genes.fasta', 
            'labels': 'elegans_labels.txt'
        },
        'D.melanogaster': {
            'fasta': 'melanogaster_genes.fasta',
            'labels': 'melanogaster_labels.txt'
        },
        'H.sapiens': {
            'fasta': 'sapiens_genes.fasta',
            'labels': 'sapiens_labels.txt'
        },
        'M.maripaludis': {
            'fasta': 'maripaludis_genes.fasta',
            'labels': 'maripaludis_labels.txt'
        },
        'S.cerevisiae': {
            'fasta': 'saccharomyces_genes.fasta',
            'labels': 'saccharomyces_labels.txt'
        }
    }
    
    # Results storage
    results = []
    
    # Base paths
    data_dir = os.path.join("..", "data", "Processed")
    models_dir = os.path.join("..", "models")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    print("🚀 Starting Cross-Species Pipeline")
    print("=" * 60)
    
    # Iterate over each training species
    for train_species, train_files in species_data.items():
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING ON: {train_species}")
        print(f"{'='*60}")
        
        # Clean memory before training
        safe_memory_cleanup()
        
        # Prepare training data paths
        train_fasta_path = os.path.join(data_dir, train_files['fasta'])
        train_label_path = os.path.join(data_dir, train_files['labels'])
        model_path = os.path.join(models_dir, f"{train_species.replace('.', '_')}_model.pt")
        
        # Train model on current species
        try:
            print(f"📂 Loading training data from {train_fasta_path}")
            data = Biodata.Biodata(
                fasta_file=str(train_fasta_path),
                label_file=str(train_label_path),
                feature_file=None,
                K=3,
                d=2
            )
            
            print("🔧 Encoding training data...")
            train_dataset = data.encode(thread=48)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"💻 Using device: {device}")
            
            model = CNmodel.model(
                label_num=2,
                other_feature_dim=0,
                K=3, 
                d=2,
                node_hidden_dim=3,
                gcn_dim=64,
                gcn_layer_num=4,
                cnn_dim=128,
                cnn_layer_num=2,
                cnn_kernel_size=8,
                fc_dim=100,
                dropout_rate=0.2,
                pnode_nn=True, 
                fnode_nn=True
            ).to(device)
            
            print("🏋️ Starting training...")
            CNmodel.train(
                train_dataset,
                model,
                learning_rate=1e-4,
                batch_size=32,
                epoch_n=200,
                random_seed=111,
                val_split=0.3,
                weighted_sampling=True,
                model_name=str(model_path)
            )
            
            print(f"✅ Model trained and saved to {model_path}")
            
        except Exception as e:
            print(f"❌ Error training on {train_species}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue
        
        # Clean up training objects
        print("🧹 Cleaning up after training...")
        if 'model' in locals():
            del model
        if 'train_dataset' in locals():
            del train_dataset
        if 'data' in locals():
            del data
        safe_memory_cleanup()
        
        # Test on all species
        for test_species, test_files in species_data.items():
            print(f"\n  🧪 Testing on: {test_species}")
            
            try:
                # Clean memory before each test
                safe_memory_cleanup()
                
                # Prepare testing data paths
                test_fasta_path = os.path.join(data_dir, test_files['fasta'])
                test_label_path = os.path.join(data_dir, test_files['labels'])
                
                # Load and encode test data
                test_data = Biodata.Biodata(
                    fasta_file=str(test_fasta_path),
                    label_file=str(test_label_path),
                    feature_file=None,
                    K=3,
                    d=2
                )
                
                print("  🔧 Encoding test data...")
                test_dataset = test_data.encode(thread=48)
                
                # Test the model
                print("  🧪 Running test...")
                test_results = CNmodel.test(test_dataset, model_name=str(model_path), val_split=1)
                
                # Add species information to results
                test_results['train_species'] = train_species
                test_results['test_species'] = test_species
                
                results.append(test_results)
                print(f"  ✅ Test completed: ACC={test_results['ACC']:.4f}, AUC={test_results['AUC']:.4f}")
                
                # Clean up test objects
                del test_dataset, test_data
                safe_memory_cleanup()
                
            except Exception as e:
                print(f"  ❌ Error testing {train_species} model on {test_species}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                
                # Add error record
                error_result = {
                    'train_species': train_species,
                    'test_species': test_species,
                    'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0,
                    'SN': 0.0, 'SP': 0.0, 'ACC': 0.0, 'AUC': 0.0,
                    'error': str(e)
                }
                results.append(error_result)
                
                # Clean up even on error
                safe_memory_cleanup()
    
    return results


def main():
    """Main function to run the pipeline"""
    print("🚀 Starting Cross-Species GNN Training and Testing Pipeline")
    print("=" * 60)
    
    # Set memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  GPU not available, using CPU")
    
    print(f"📊 Total species: 6")
    print(f"🎯 Training on all species")
    print(f"🧪 Testing on all species")
    print("=" * 60)
    
    try:
        # Run the pipeline
        results = train_test_pipeline()
        
        # Save results
        if results:
            df = save_results_to_excel(results)
            
            # Print final summary
            successful_tests = len([r for r in results if 'error' not in r or not r['error']])
            failed_tests = len([r for r in results if 'error' in r and r['error']])
            
            print(f"\n🎉 PIPELINE COMPLETED!")
            print(f"✅ Successful tests: {successful_tests}")
            print(f"❌ Failed tests: {failed_tests}")
            print(f"📊 Total results: {len(results)}")
            
            if df is not None:
                print(f"\n📈 Performance Summary:")
                print(f"Average ACC: {df['ACC'].mean():.4f}")
                print(f"Average AUC: {df['AUC'].mean():.4f}")
                print(f"Best ACC: {df['ACC'].max():.4f}")
                print(f"Best AUC: {df['AUC'].max():.4f}")
        
        else:
            print("❌ No results generated from pipeline!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n💥 Fatal error in pipeline: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        # Final cleanup
        safe_memory_cleanup()
        print("\n🧹 Final memory cleanup completed")


if __name__ == "__main__":
    main()