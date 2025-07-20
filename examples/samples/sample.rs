// examples/samples/sample.rs
//! Sample Rust File
//! =================
//! 
//! Demonstrates various Rust language features for semantic chunking:
//! - Traits and implementations
//! - Async/await with tokio
//! - Error handling with Result types
//! - Generics and lifetimes
//! - Pattern matching
//! - Ownership and borrowing

use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use anyhow::{Context, Result};

/// Configuration for data processing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub batch_size: usize,
    pub timeout: Duration,
    pub retries: u32,
    pub debug: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            timeout: Duration::from_secs(30),
            retries: 3,
            debug: false,
        }
    }
}

impl ProcessingConfig {
    /// Create a new configuration with custom batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable debug mode
    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(anyhow::anyhow!("Batch size must be greater than 0"));
        }
        if self.timeout.is_zero() {
            return Err(anyhow::anyhow!("Timeout must be greater than 0"));
        }
        Ok(())
    }
}

/// Trait defining the interface for data processors
#[async_trait::async_trait]
pub trait DataProcessor<T, R> {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Process a single item
    async fn process_item(&self, item: T) -> Result<R, Self::Error>;

    /// Process multiple items in batch
    async fn process_batch(&self, items: Vec<T>) -> Result<Vec<R>, Self::Error> {
        let mut results = Vec::with_capacity(items.len());
        
        for item in items {
            match self.process_item(item).await {
                Ok(result) => results.push(result),
                Err(e) => return Err(e),
            }
        }
        
        Ok(results)
    }

    /// Get processor statistics
    fn get_stats(&self) -> ProcessorStats;
}

/// Statistics for data processing operations
#[derive(Debug, Default, Clone, Serialize)]
pub struct ProcessorStats {
    pub processed: u64,
    pub errors: u64,
    pub retries: u64,
    pub total_time: Duration,
}

impl Display for ProcessorStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Stats: {} processed, {} errors, {} retries, {:.2}s total",
            self.processed,
            self.errors,
            self.retries,
            self.total_time.as_secs_f64()
        )
    }
}

/// Data structure representing a processing item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    pub id: u64,
    pub name: String,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

impl DataItem {
    /// Create a new data item
    pub fn new(id: u64, name: impl Into<String>, value: f64) -> Self {
        Self {
            id,
            name: name.into(),
            value,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the item
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if item is valid
    pub fn is_valid(&self) -> bool {
        !self.name.is_empty() && self.value.is_finite()
    }
}

/// Result type for processed data
#[derive(Debug, Clone, Serialize)]
pub struct ProcessedData {
    pub original_id: u64,
    pub processed_value: f64,
    pub timestamp: u64,
    pub processor_id: String,
}

/// Error types for data processing
#[derive(thiserror::Error, Debug)]
pub enum ProcessingError {
    #[error("Invalid data: {message}")]
    InvalidData { message: String },
    
    #[error("Processing timeout after {seconds} seconds")]
    Timeout { seconds: u64 },
    
    #[error("Network error: {source}")]
    Network { 
        #[from]
        source: std::io::Error 
    },
    
    #[error("Serialization error: {source}")]
    Serialization {
        #[from]
        source: serde_json::Error
    },
}

/// Asynchronous data processor implementation
pub struct AsyncDataProcessor {
    config: ProcessingConfig,
    stats: Arc<Mutex<ProcessorStats>>,
    processor_id: String,
}

impl AsyncDataProcessor {
    /// Create a new async data processor
    pub fn new(config: ProcessingConfig) -> Result<Self> {
        config.validate().context("Invalid configuration")?;
        
        Ok(Self {
            config,
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
            processor_id: uuid::Uuid::new_v4().to_string(),
        })
    }

    /// Create processor with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(ProcessingConfig::default())
    }

    /// Process items with retry logic
    async fn process_with_retry(&self, item: DataItem, attempt: u32) -> Result<ProcessedData, ProcessingError> {
        match self.process_single_item(item.clone()).await {
            Ok(result) => Ok(result),
            Err(e) if attempt < self.config.retries => {
                // Update retry stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.retries += 1;
                }

                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                if self.config.debug {
                    eprintln!("Retry {} for item {} after {:?}", attempt + 1, item.id, delay);
                }
                
                sleep(delay).await;
                self.process_with_retry(item, attempt + 1).await
            }
            Err(e) => {
                // Update error stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.errors += 1;
                }
                Err(e)
            }
        }
    }

    /// Process a single item (internal implementation)
    async fn process_single_item(&self, item: DataItem) -> Result<ProcessedData, ProcessingError> {
        if !item.is_valid() {
            return Err(ProcessingError::InvalidData {
                message: format!("Item {} is invalid", item.id),
            });
        }

        // Simulate async processing
        let processing_time = Duration::from_millis(10 + (item.id % 50));
        
        tokio::select! {
            _ = sleep(processing_time) => {
                // Successful processing
                let processed_value = item.value * 2.0 + (item.id as f64).sqrt();
                
                Ok(ProcessedData {
                    original_id: item.id,
                    processed_value,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    processor_id: self.processor_id.clone(),
                })
            }
            _ = sleep(self.config.timeout) => {
                Err(ProcessingError::Timeout {
                    seconds: self.config.timeout.as_secs(),
                })
            }
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &ProcessingConfig {
        &self.config
    }

    /// Get processor ID
    pub fn id(&self) -> &str {
        &self.processor_id
    }
}

#[async_trait::async_trait]
impl DataProcessor<DataItem, ProcessedData> for AsyncDataProcessor {
    type Error = ProcessingError;

    async fn process_item(&self, item: DataItem) -> Result<ProcessedData, Self::Error> {
        let start_time = std::time::Instant::now();
        
        let result = self.process_with_retry(item, 0).await;
        
        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.processed += 1;
            stats.total_time += start_time.elapsed();
        }

        result
    }

    async fn process_batch(&self, items: Vec<DataItem>) -> Result<Vec<ProcessedData>, Self::Error> {
        let mut results = Vec::with_capacity(items.len());
        
        // Process in configurable batch sizes
        for chunk in items.chunks(self.config.batch_size) {
            let batch_futures: Vec<_> = chunk
                .iter()
                .cloned()
                .map(|item| self.process_item(item))
                .collect();
            
            // Wait for all items in this batch
            let batch_results = futures::future::try_join_all(batch_futures).await?;
            results.extend(batch_results);
        }
        
        Ok(results)
    }

    fn get_stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap_or_else(|_| {
            panic!("Stats mutex poisoned")
        }).clone()
    }
}

/// Generic data storage trait
pub trait DataStorage<T> {
    type Key;
    type Error: std::error::Error;

    fn store(&mut self, key: Self::Key, data: T) -> Result<(), Self::Error>;
    fn retrieve(&self, key: &Self::Key) -> Result<Option<T>, Self::Error>;
    fn delete(&mut self, key: &Self::Key) -> Result<bool, Self::Error>;
    fn list_keys(&self) -> Result<Vec<Self::Key>, Self::Error>;
}

/// In-memory storage implementation
pub struct MemoryStorage<T> {
    data: HashMap<String, T>,
}

impl<T> MemoryStorage<T> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> Default for MemoryStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> DataStorage<T> for MemoryStorage<T> {
    type Key = String;
    type Error = std::io::Error;

    fn store(&mut self, key: Self::Key, data: T) -> Result<(), Self::Error> {
        self.data.insert(key, data);
        Ok(())
    }

    fn retrieve(&self, key: &Self::Key) -> Result<Option<T>, Self::Error> {
        Ok(self.data.get(key).cloned())
    }

    fn delete(&mut self, key: &Self::Key) -> Result<bool, Self::Error> {
        Ok(self.data.remove(key).is_some())
    }

    fn list_keys(&self) -> Result<Vec<Self::Key>, Self::Error> {
        Ok(self.data.keys().cloned().collect())
    }
}

/// Utility functions for data manipulation
pub mod utils {
    use super::*;

    /// Filter and transform data items
    pub fn filter_and_transform<F, T>(items: Vec<DataItem>, predicate: F) -> Vec<T>
    where
        F: Fn(&DataItem) -> Option<T>,
    {
        items.iter().filter_map(predicate).collect()
    }

    /// Group items by a key function
    pub fn group_by<K, F>(items: Vec<DataItem>, key_fn: F) -> HashMap<K, Vec<DataItem>>
    where
        K: Eq + std::hash::Hash,
        F: Fn(&DataItem) -> K,
    {
        let mut groups: HashMap<K, Vec<DataItem>> = HashMap::new();
        
        for item in items {
            let key = key_fn(&item);
            groups.entry(key).or_default().push(item);
        }
        
        groups
    }

    /// Calculate statistics for a collection of items
    pub fn calculate_stats(items: &[DataItem]) -> (f64, f64, f64) {
        if items.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let sum: f64 = items.iter().map(|item| item.value).sum();
        let count = items.len() as f64;
        let mean = sum / count;

        let min = items.iter().map(|item| item.value).fold(f64::INFINITY, f64::min);
        let max = items.iter().map(|item| item.value).fold(f64::NEG_INFINITY, f64::max);

        (mean, min, max)
    }
}

/// Main demonstration function
#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Running Rust demo...");

    // Create configuration
    let config = ProcessingConfig::default()
        .with_batch_size(10)
        .with_debug();

    // Create processor
    let processor = AsyncDataProcessor::new(config)
        .context("Failed to create processor")?;

    // Create test data
    let test_data: Vec<DataItem> = (1..=25)
        .map(|i| {
            DataItem::new(i, format!("item_{}", i), i as f64 * 3.14)
                .with_metadata("category", "test")
                .with_metadata("batch", "demo")
        })
        .collect();

    println!("📊 Processing {} items...", test_data.len());

    // Process data
    match processor.process_batch(test_data.clone()).await {
        Ok(results) => {
            println!("✅ Processed {} items successfully", results.len());
            
            // Show some results
            for (i, result) in results.iter().take(3).enumerate() {
                println!("  {}. ID: {}, Value: {:.2}", 
                    i + 1, result.original_id, result.processed_value);
            }
            
            if results.len() > 3 {
                println!("  ... and {} more", results.len() - 3);
            }
        }
        Err(e) => {
            eprintln!("❌ Processing failed: {}", e);
            return Err(e.into());
        }
    }

    // Show statistics
    let stats = processor.get_stats();
    println!("📈 {}", stats);

    // Demonstrate utility functions
    let filtered = utils::filter_and_transform(test_data.clone(), |item| {
        if item.value > 15.0 {
            Some(item.id)
        } else {
            None
        }
    });
    println!("🔍 {} items with value > 15.0", filtered.len());

    let (mean, min, max) = utils::calculate_stats(&test_data);
    println!("📊 Stats - Mean: {:.2}, Min: {:.2}, Max: {:.2}", mean, min, max);

    // Test storage
    let mut storage = MemoryStorage::new();
    for item in test_data.iter().take(5) {
        storage.store(item.id.to_string(), item.clone())?;
    }
    println!("💾 Stored {} items in memory", storage.len());

    println!("🎉 Demo completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_item_creation() {
        let item = DataItem::new(1, "test", 42.0)
            .with_metadata("key", "value");
        
        assert_eq!(item.id, 1);
        assert_eq!(item.name, "test");
        assert_eq!(item.value, 42.0);
        assert!(item.is_valid());
        assert_eq!(item.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_config_validation() {
        let valid_config = ProcessingConfig::default();
        assert!(valid_config.validate().is_ok());

        let invalid_config = ProcessingConfig {
            batch_size: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[tokio::test]
    async fn test_processor_creation() {
        let config = ProcessingConfig::default();
        let processor = AsyncDataProcessor::new(config);
        assert!(processor.is_ok());
    }
}