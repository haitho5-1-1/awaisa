# Crypto Trading Analytics Dashboard

## Overview

This is a Streamlit-based cryptocurrency trading analytics dashboard that integrates multiple data sources to provide comprehensive market insights. The application combines technical analysis, sentiment analysis, on-chain metrics, and machine learning predictions to help users make informed trading decisions.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Interface**: Single-page application with interactive widgets and real-time data visualization
- **Deployment**: Streamlit Cloud compatible with secrets management

### Backend Architecture
- **Language**: Python 3.x
- **Data Processing**: Pandas and NumPy for data manipulation
- **Machine Learning**: Scikit-learn with Gaussian Process Classifier for predictions
- **API Integration**: RESTful API calls using requests library

### Data Sources Integration
The application integrates with multiple cryptocurrency and social media APIs:
- **Market Data**: Bitget API for OHLCV and order book data
- **Social Sentiment**: Twitter API (via Tweepy) and Reddit API (via PRAW)
- **Search Trends**: Google Trends (via PyTrends)
- **Traditional Markets**: Yahoo Finance (via yfinance)
- **On-chain Analytics**: Dune Analytics, Santiment, The Graph
- **News & Events**: CryptoPanic, CoinMarketCal
- **Social Analytics**: LunarCrush
- **Research Data**: Messari API

## Key Components

### 1. Configuration Management
- Centralized API key management using Streamlit secrets
- Graceful handling of missing API keys with fallback values
- Environment-based configuration for different deployment stages

### 2. Data Fetching Layer
- **Market Data**: Real-time OHLCV data and order book information from Bitget
- **Sentiment Analysis**: Social media data processing with TextBlob for sentiment scoring
- **Multi-source Aggregation**: Unified data fetching functions for various APIs

### 3. Machine Learning Pipeline
- **Algorithm**: Gaussian Process Classifier for pattern recognition
- **Kernels**: RBF and Constant kernels for flexible modeling
- **Preprocessing**: StandardScaler for feature normalization
- **Real-time Predictions**: Live market condition classification

### 4. Analytics Engine
- Technical indicators calculation
- Sentiment aggregation and scoring
- On-chain metrics analysis
- Multi-timeframe analysis capabilities

## Data Flow

1. **Data Ingestion**: APIs are called to fetch real-time market data, social sentiment, and on-chain metrics
2. **Data Processing**: Raw data is cleaned, normalized, and transformed using pandas operations
3. **Feature Engineering**: Technical indicators, sentiment scores, and derived metrics are calculated
4. **Machine Learning**: Gaussian Process models analyze patterns and generate predictions
5. **Visualization**: Processed data is displayed through Streamlit's interactive components
6. **Real-time Updates**: Dashboard refreshes data based on user interactions and time intervals

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **requests**: HTTP API calls

### Social Media & Sentiment
- **tweepy**: Twitter API integration
- **praw**: Reddit API integration
- **textblob**: Natural language processing for sentiment analysis
- **pytrends**: Google Trends data access

### Financial Data
- **yfinance**: Yahoo Finance data access

### API Integrations
- Bitget API for cryptocurrency market data
- Twitter API for social sentiment
- Reddit API for community discussions
- Dune Analytics for on-chain data
- Santiment for crypto analytics
- The Graph for blockchain data
- CryptoPanic for news aggregation
- LunarCrush for social analytics
- Messari for research data
- CoinMarketCal for events

## Deployment Strategy

### Platform
- **Primary**: Streamlit Cloud deployment
- **Configuration**: Secrets management through Streamlit's built-in system
- **Scalability**: Stateless application design for easy horizontal scaling

### Environment Management
- API keys stored securely in Streamlit secrets
- Graceful degradation when APIs are unavailable
- Error handling and user feedback for failed API calls

### Performance Considerations
- Caching strategies for expensive API calls
- Efficient data processing with vectorized operations
- Responsive UI design for real-time updates

## Changelog

```
Changelog:
- June 30, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```