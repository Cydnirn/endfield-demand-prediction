## Prerequisites
- AWS Account with administrative permissions (or sufficient rights to create VPC, EMR, SageMaker, etc.).
- AWS Console access – you will perform all deployments manually via the browser (or optionally use AWS CLI for some steps).
- A sample CSV file with sales data (provided in /data).
- Basic knowledge of AWS services (S3, EMR, SageMaker, Step Functions, API Gateway, Lambda, EventBridge).

## Step‑by‑Step Deployment Guide (Console‑Focused)

### Phase 1: Prepare the S3 Buckets & Script

1. **Create the data bucket** (e.g., `endfield-data-12345`) and the scripts bucket (`endfield-scripts-12345`).
    
2. Upload the PySpark script (`transform.py`) to the scripts bucket root.
    
3. Upload a sample CSV file to the data bucket under `/data/` (this will be used later to trigger the pipeline).
    

---

### Phase 2: Set Up the EMR Cluster

1. In the EMR console, click **Create cluster**.
    
2. Enter `endfield-emr-cluster` as the name.
    
3. Choose **EMR release 6.9.0**.
    
4. Select **Spark** and **Hadoop** applications.
    
5. Under **Instance groups**, set 1 master (m5.xlarge) and 1 core (m5.xlarge).
    
6. Under **Security and access**, choose the IAM roles you created (`endfield-emr-ec2-role` and `endfield-emr-service-role`).
    
7. Enable logging to an S3 location (create a log bucket if needed).
    
8. Launch the cluster. Wait for it to reach **Waiting** state.
    

---

### Phase 3: Create the Step Functions State Machine

1. Create a new state machine based on the requirements from the document. Ensure the state machine has graceful error handling and retries.
    
2. Select **Create a new role** or use an existing role with required permissions.
    
3. Name it `endfield-forecast-pipeline` and create.

---

### Phase 4: Configure SageMaker Endpoint

1. Name: `endfield-forecast-endpoint`.
    
2. For the endpoint configuration, choose **Create a new endpoint configuration**.
    
3. Select **Add model** → **Create a new model** with a dummy XGBoost model (you can use a sample model from the built‑in algorithm or just a placeholder – the pipeline will update it later).
    
4. Choose instance type `ml.m5.xlarge` and deploy. Wait for **InService**.
    

---

### Phase 5: Create the Lambda Function
    
1. Name: `endfield-forecast-lambda`, runtime Python 3.9.
    
2. Under **Permissions**, choose an IAM role with `sagemaker:InvokeEndpoint` and `logs:CreateLogGroup` etc.
    
3. Set environment variable `ENDPOINT_NAME` = `endfield-forecast-endpoint`.
  
---

### Phase 6: Create API Gateway and Link Lambda

1. In API Gateway, create a **REST API** (not HTTP).
    
2. Name: `endfield-forecast-api`.
    
3. Create a resource `/predict` and a `POST` method.
    
4. Integration type: Lambda, select the `endfield-forecast-lambda` function.
    
5. Enable CORS (enable `Access-Control-Allow-Origin: '*'` for testing).
    
6. Deploy the API to a stage (e.g., `prod`). Note the invoke URL – you will need it for testing.
    

---

### Phase 7: Set Up EventBridge Rule
    
1. Name: `endfield-s3-upload-trigger`.
    
2. Define event pattern:
    
    - Source: AWS services → S3.
        
    - Detail type: Object Created.
        
    - Under **Specify bucket(s) by name**, add your data bucket.
        
    - Under **Object key** → **Prefix**, enter `data/`.
        
3. Target: Step Functions state machine, select `endfield-forecast-pipeline`.
    
---

### Phase 8: Test the End‑to‑End Pipeline

1. Upload a new sample CSV file to your data bucket under `/data/` (use AWS CLI or Console).
    
2. Immediately go to Step Functions console – you should see a new execution start.
    
3. Watch the visualiser – it will go through EMR step submission, wait for completion, start SageMaker training, and finally update the endpoint.
    
To interact with the forecasting API via a user‑friendly web interface, you can deploy the provided static files (HTML, CSS, JS) to AWS Amplify or S3.

**Option A: AWS Amplify (recommended for simplicity)**

1. In the Amplify console, click **Host web app**.
    
2. Choose **Deploy without Git** and upload the folder containing `index.html`, `index.js`, and `style.css`.
    
3. Amplify will provide a public URL (e.g., `https://main.<random>.amplifyapp.com`).
    
4. Open the URL, enter the required fields, and click **Get Forecast** – the page will call your API and display the results.
    

**Option B: S3 Static Website**

1. Create an S3 bucket (e.g., `endfield-frontend`), enable **Static website hosting**, and set the index document to `index.html`.
    
2. Upload the static files to the bucket.
    
3. Add a bucket policy that allows public read access (for testing).
    
4. Access the bucket endpoint URL (in the bucket properties) to use the app.
    

**Note**: Ensure your API Gateway has CORS enabled for the origin of your frontend (or allow all origins with `*` for testing). The frontend code should point to the API Gateway invoke URL – you can hardcode it or read it from a configuration.
