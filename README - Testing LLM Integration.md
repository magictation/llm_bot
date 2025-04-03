# LLM Integration Testing for Bitcoin Trading Bot

This document provides instructions for testing the LLM (Google Generative AI) integration in your Bitcoin Trading Signal Bot. The test scripts will verify that the LLM component is working correctly before running the bot in real-time mode.

## Test Scripts Overview

I've created two test scripts:

1. **test_llm.py** - Basic test that verifies API connectivity and response format
2. **test_llm_with_real_data.py** - Advanced test using real Bitcoin market data from Binance

## Prerequisites

Before running the tests, make sure:

1. Your `.env` file contains the `GEMINI_API_KEY` variable with your Google Generative AI API key
2. You have internet access for API calls and Binance data retrieval
3. All required dependencies are installed

## Running the Basic Test

This test verifies:
- API key configuration
- Simple prompt functionality
- Market analysis with sample data
- JSON parsing and validation

```bash
python test_llm.py
```

If successful, you should see a "All tests passed!" message at the end.

## Running the Real Data Test

This test fetches actual Bitcoin data from Binance and runs a complete market analysis using the LLM:

```bash
python test_llm_with_real_data.py
```

This test simulates exactly what your trading bot will do in real-time mode, including:
- Fetching data from Binance
- Adding technical indicators
- Sending data to the LLM
- Processing the response

## What to Check For

For both tests, verify:

1. **Complete JSON Structure**: The response should contain all required fields:
   - market_condition
   - signal (LONG/SHORT/NEUTRAL)
   - confidence (0-100%)
   - support_levels
   - resistance_levels
   - stop_loss
   - take_profit
   - risk_reward_ratio
   - reasoning

2. **Response Quality**: The analysis should be coherent and match the market data

3. **Response Time**: Note how long the LLM takes to respond (typically 5-15 seconds)

## Troubleshooting

If the tests fail, check:

1. **API Key**: Verify your Gemini API key is correct and has sufficient quota
2. **Network Connectivity**: Ensure you can reach the Google Generative AI API
3. **Error Logs**: Check the error messages for specific issues

## Next Steps

Once both tests pass successfully, you can be confident that:

1. Your LLM integration is working correctly
2. The bot will be able to utilize AI analysis in real-time mode
3. The response format is compatible with your trading bot

You're now ready to run your bot in real-time mode tomorrow with the assurance that the LLM component will function as expected!