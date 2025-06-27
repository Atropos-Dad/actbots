# GitHub Secrets Setup for Lambda Auto-Deployment

To enable automatic deployment of your Lambda function, you need to configure these secrets in your GitHub repository:

## How to Add Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each of the following:

## Required Secrets

### AWS Credentials
- **`AWS_ACCESS_KEY_ID`**: Your AWS access key ID
- **`AWS_SECRET_ACCESS_KEY`**: Your AWS secret access key  
- **`AWS_REGION`**: AWS region where your Lambda is deployed (e.g., `us-east-1`)

### Lambda Configuration
- **`S3_LAMBDA_BUCKET`**: S3 bucket name where your Lambda package is stored
- **`S3_LAMBDA_KEY`**: S3 object key/filename (e.g., `discord-bot-lambda.zip`)
- **`LAMBDA_FUNCTION_NAME`**: Your Lambda function name (e.g., `discord-bot-handler`)

## AWS IAM Permissions

The AWS user needs these permissions (attach these policies):

### Required Policies:
1. **AmazonS3FullAccess** (or more restrictive S3 bucket permissions)
2. **AWSLambda_FullAccess** (or more restrictive Lambda permissions)

### Custom Policy (More Secure):
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "lambda:UpdateFunctionCode"
            ],
            "Resource": "arn:aws:lambda:YOUR_REGION:YOUR_ACCOUNT_ID:function:YOUR_FUNCTION_NAME"
        }
    ]
}
```

## Initial Setup

Before the automated deployment can work, you need to:

1. **Create the S3 bucket** and upload an initial Lambda package
2. **Create the Lambda function** pointing to the S3 package
3. **Set up the GitHub secrets** as listed above

After this one-time setup, every commit to `main` will automatically update your Lambda function!

## Monitoring Deployments

- Check the **Actions** tab in your GitHub repository to see deployment status
- View AWS CloudWatch logs for your Lambda function to verify updates
- The workflow will show which files were changed and deployed

## Security Best Practices

1. Use least-privilege IAM policies
2. Consider using AWS IAM roles with OIDC instead of access keys
3. Regularly rotate your AWS credentials
4. Monitor CloudTrail for deployment activities 