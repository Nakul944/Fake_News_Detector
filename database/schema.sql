CREATE DATABASE IF NOT EXISTS fake_news_detector;

USE fake_news_detector;

CREATE TABLE IF NOT EXISTS prediction_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    claim TEXT NOT NULL,
    predicted_label VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

