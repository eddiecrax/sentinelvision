#!/bin/bash
# SentinelVision — AWS ECS/Fargate Deployment Automation Script
set -e

REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-<YOUR_AWS_ACCOUNT_ID>}"
ECR_REPO="${ECR_REPO:-sentinelvision}"
ARCH="${ARCH:-x86_64}" # Options: x86_64, arm64
if [ "$ARCH" = "arm64" ]; then
    IMAGE_TAG="${IMAGE_TAG:-arm64}"
    TASK_DEF="deployment/aws/task-definition-arm64.json"
    DOCKER_PLATFORM_FLAG="--platform linux/arm64"
else
    IMAGE_TAG="${IMAGE_TAG:-latest}"
    TASK_DEF="deployment/aws/task-definition.json"
    DOCKER_PLATFORM_FLAG="--platform linux/amd64"
fi
ENABLE_AWS_PUSH="${ENABLE_AWS_PUSH:-false}"

echo "============================================================"
echo "  SENTINELVISION — AWS ECS / Fargate Container Deployment"
echo "  Target Region: $REGION"
echo "============================================================"

# 1. Build Local Docker Container
echo "[1/4] Building Docker Image: $ECR_REPO:$IMAGE_TAG..."
docker build $DOCKER_PLATFORM_FLAG -t $ECR_REPO:$IMAGE_TAG -f deployment/Dockerfile .

# 2. Local Container Test Check
echo "[2/4] Testing container health check..."
docker run --rm -d --name sentinelvision-test -p 8000:8000 $ECR_REPO:$IMAGE_TAG
sleep 5
curl -f http://localhost:8000/api/health || { echo "❌ Healthcheck failed!"; docker stop sentinelvision-test; exit 1; }
docker stop sentinelvision-test
echo "✅ Local Container Test Passed!"

# 3. AWS ECR Login and Push (Only when explicitly enabled and account ID configured)
if [ "$ENABLE_AWS_PUSH" = "true" ] && [ "$AWS_ACCOUNT_ID" != "<YOUR_AWS_ACCOUNT_ID>" ]; then
    ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
    echo "[3/4] Tagging image for ECR: $ECR_URI..."
    docker tag $ECR_REPO:$IMAGE_TAG $ECR_URI

    if command -v aws &> /dev/null; then
        echo "Attempting AWS ECR Login..."
        aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
        echo "Pushing image to AWS ECR..."
        docker push $ECR_URI
    else
        echo "❌ AWS CLI not found; cannot push to remote ECR."
        exit 1
    fi

    # 4. Register ECS Task Definition
    echo "[4/4] Registering ECS Task Definition ($TASK_DEF)..."
    aws ecs register-task-definition --cli-input-json file://$TASK_DEF --region $REGION
else
    echo "[3/4] Skipping AWS ECR Push (ENABLE_AWS_PUSH not set to true or AWS_ACCOUNT_ID is placeholder)."
    echo "[4/4] Skipping ECS Task Registration (Local validation mode)."
fi

echo "============================================================"
echo "  ✅ SENTINELVISION AWS DEPLOYMENT BUILD COMPLETED!"
echo "============================================================"
