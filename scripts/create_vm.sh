export AIMSUSERNAME="similoluwa"
export REGION="europe-west4"
export ZONE="europe-west4-a"

# Create VPC if it doesn't exist
if ! gcloud compute networks describe ${AIMSUSERNAME}-vpc &>/dev/null; then
    gcloud compute networks create ${AIMSUSERNAME}-vpc --subnet-mode=auto
else
    echo "VPC ${AIMSUSERNAME}-vpc already exists, skipping."
fi

# Create firewall rule if it doesn't exist
if ! gcloud compute firewall-rules describe ${AIMSUSERNAME}-fw-ssh &>/dev/null; then
    gcloud compute firewall-rules create ${AIMSUSERNAME}-fw-ssh \
        --network=${AIMSUSERNAME}-vpc \
        --allow=tcp:22
else
    echo "Firewall rule ${AIMSUSERNAME}-fw-ssh already exists, skipping."
fi

# Create Cloud Router if it doesn't exist
if ! gcloud compute routers describe ${AIMSUSERNAME}-router-${REGION} --region=${REGION} &>/dev/null; then
    gcloud compute routers create ${AIMSUSERNAME}-router-${REGION} \
        --network=${AIMSUSERNAME}-vpc \
        --region=${REGION}
else
    echo "Router ${AIMSUSERNAME}-router-${REGION} already exists, skipping."
fi

# Create NAT if it doesn't exist
if ! gcloud compute routers nats describe ${AIMSUSERNAME}-nat-config \
    --router=${AIMSUSERNAME}-router-${REGION} \
    --region=${REGION} &>/dev/null; then
    gcloud compute routers nats create ${AIMSUSERNAME}-nat-config \
        --router-region=${REGION} \
        --router=${AIMSUSERNAME}-router-${REGION} \
        --nat-all-subnet-ip-ranges \
        --auto-allocate-nat-external-ips
else
    echo "NAT ${AIMSUSERNAME}-nat-config already exists, skipping."
fi 

export VM_NAME="${AIMSUSERNAME}-l4-vm"

gcloud compute instances create ${VM_NAME} \
    --zone=${ZONE} \
    --machine-type=g2-standard-4 \
    --accelerator="type=nvidia-l4,count=1" \
    --image-family=common-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --network=${AIMSUSERNAME}-vpc \
    --scopes=storage-full,cloud-platform