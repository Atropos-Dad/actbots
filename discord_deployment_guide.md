# Discord Bot Lambda Deployment Guide

This guide walks you through deploying your ActBots AI agent as a Discord bot using AWS Lambda.

## Prerequisites

1. **Discord Developer Account**: [Discord Developer Portal](https://discord.com/developers/applications)
2. **AWS Account**: With Lambda and API Gateway access
3. **API Keys**: OpenAI and Jentic platform access

## Step 1: Discord Bot Setup

### 1.1 Create Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Name your application (e.g., "ActBots AI Agent")
4. Save the **Application ID** (you'll need this later)

### 1.2 Create Bot User

1. Go to the "Bot" tab in your application
2. Click "Add Bot"
3. Save the **Bot Token** (keep this secret!)
4. Under "Privileged Gateway Intents", you can leave these disabled for slash commands

### 1.3 Get Public Key

1. Go to "General Information" tab
2. Copy the **Public Key** (needed for signature verification)

## Step 2: Environment Setup

### 2.1 Create Environment File

Create a `.env` file in your project root:

```bash
# Discord Configuration
DISCORD_APP_ID=your_application_id_here
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_PUBLIC_KEY=your_public_key_here
DISCORD_GUILD_ID=your_test_server_id_here  # Optional: for faster command updates

# API Keys
JENTIC_API_KEY=your_jentic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo  # Optional: defaults to gpt-4-turbo

# AWS Configuration (if deploying via CLI)
AWS_REGION=us-east-1
AWS_PROFILE=default
```

### 2.2 Get Discord Guild ID (Optional)

For faster command testing:
1. Enable Developer Mode in Discord (User Settings > Advanced > Developer Mode)
2. Right-click your server name → "Copy Server ID"
3. Add this as `DISCORD_GUILD_ID` in your `.env` file

## Step 3: Register Discord Commands

### 3.1 Install Dependencies

For running the registration script, you'll need the `requests` library.

```bash
pip install requests
```

### 3.2 Register Commands

```bash
# Load environment variables
source .env  # or use your preferred method

# Register commands
python discord_commands/register_commands.py
```

The script will register these commands:
- `/ask` - Ask the AI agent a question
- `/help` - Get help information
- `/search` - Search for available tools
- `/status` - Check agent status

## Step 4: Package Lambda Function

This step creates the `.zip` file for your Lambda function. By using the smaller `requirements.txt`, the package size will be significantly reduced.

### 4.1 Create Deployment Package

```bash
# Create a deployment directory
mkdir lambda_package

# Copy your code
cp -r jentic_agents lambda_package/
cp lambda_discord_bot.py lambda_package/

# Install ONLY production dependencies into the package
pip install -r requirements.txt --target lambda_package/

# Change into the package directory
cd lambda_package

# (Optional but Recommended) Clean up to reduce size
echo "Cleaning up package to reduce size..."
find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf *.dist-info
rm -rf *__pycache__*

# Create the final deployment package
zip -r ../discord_bot_lambda.zip .

# Go back to the root directory
cd ..
```

### 4.2 Alternative: Using Docker for Lambda

Create a `Dockerfile`:

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copy application code
COPY jentic_agents/ ${LAMBDA_TASK_ROOT}/jentic_agents/
COPY lambda_discord_bot.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_discord_bot.lambda_handler"]
```

Build and deploy:
```bash
docker build -t discord-bot .
# Tag and push to ECR, then create Lambda function from container image
```

## Step 5: Deploy Lambda Function

### 5.1 Create Lambda Function (AWS Console)

1. Go to AWS Lambda Console
2. Click "Create function"
3. Choose "Author from scratch"
4. Function name: `discord-bot-handler`
5. Runtime: Python 3.11
6. Architecture: `x86_64` or `arm64`
7. Upload your `discord_bot_lambda.zip` file

### 5.2 Configure Lambda

**Environment Variables:**
```
DISCORD_PUBLIC_KEY=your_public_key
DISCORD_APP_ID=your_app_id
JENTIC_API_KEY=your_jentic_key
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4-turbo
```

**Function Configuration:**
- Handler: `lambda_discord_bot.lambda_handler`
- Timeout: 15 seconds (Discord has 3-second initial response limit, but we handle this)
- Memory: 512 MB (adjust based on your needs)

### 5.3 Create API Gateway Trigger

1. In Lambda function console, click "Add trigger"
2. Select "API Gateway"
3. Create a new API:
   - API type: REST API
   - Security: Open
4. Note the API endpoint URL (you'll need this for Discord)

## Step 6: Configure Discord Interactions Endpoint

### 6.1 Set Interactions Endpoint URL

1. Go back to Discord Developer Portal
2. Go to "General Information" tab
3. Find "Interactions Endpoint URL"
4. Enter your API Gateway endpoint URL
5. Click "Save Changes"

Discord will send a verification request. If everything is set up correctly, it should save successfully.

## Step 7: Add Bot to Server

### 7.1 Generate Invite URL

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=applications.commands
```

Replace `YOUR_APP_ID` with your Discord Application ID.

### 7.2 Invite Bot

1. Open the invite URL
2. Select your server
3. Authorize the bot

## Step 8: Test the Bot

### 8.1 Test Commands

In your Discord server, try:
- `/ask question:What is 2+2?`
- `/help`
- `/status`
- `/search query:weather`

### 8.2 Monitor Logs

Check AWS CloudWatch logs for your Lambda function to debug any issues.

## Troubleshooting

### Common Issues

1. **"Invalid request signature"**
   - Check that `DISCORD_PUBLIC_KEY` is correct
   - Ensure the environment variable is set in Lambda

2. **"This interaction failed"**
   - Check Lambda logs in CloudWatch
   - Verify all environment variables are set
   - Check that dependencies are included in deployment package

3. **Commands not appearing**
   - Global commands take up to 1 hour to propagate
   - Use guild commands (set `DISCORD_GUILD_ID`) for instant updates
   - Re-run the command registration script

4. **Lambda timeout**
   - Increase Lambda timeout (max 15 minutes)
   - Consider optimizing your reasoning loop
   - Check that all API keys are valid

### Debugging Tips

1. **Test Lambda locally:**
   ```bash
   # Create a test event file
   echo '{"body": "{\"type\": 1}", "headers": {}}' > test_event.json
   
   # Test the function
   python -c "
   import json
   from lambda_discord_bot import lambda_handler
   with open('test_event.json') as f:
       event = json.load(f)
   print(lambda_handler(event, None))
   "
   ```

2. **Check Discord payload:**
   Add logging to see what Discord is sending:
   ```python
   logger.info(f"Received body: {event.get('body')}")
   ```

3. **Verify environment variables:**
   ```python
   logger.info(f"Environment check - APP_ID: {bool(DISCORD_APP_ID)}, PUBLIC_KEY: {bool(DISCORD_PUBLIC_KEY)}")
   ```

## Cost Optimization

- Use Provisioned Concurrency if you need faster cold starts
- Consider using Lambda ARM architecture for cost savings
- Monitor CloudWatch costs and set up billing alerts
- Use Lambda Power Tools for better observability

## Security Best Practices

1. **Never commit secrets to version control**
2. **Use AWS Secrets Manager for production secrets**
3. **Enable CloudTrail logging**
4. **Use least-privilege IAM roles**
5. **Regularly rotate API keys**

## Next Steps

Once your bot is working:

1. **Add more commands** by updating `discord_commands/register_commands.py`
2. **Customize responses** in the Discord agent
3. **Add persistent memory** by replacing `ScratchPadMemory`
4. **Monitor usage** and set up alerts
5. **Scale up** by adding more Jentic workflows

## Support

If you encounter issues:
1. Check AWS CloudWatch logs
2. Verify Discord Developer Portal settings
3. Test each component individually
4. Review this guide for missing steps 