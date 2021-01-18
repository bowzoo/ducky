include env.sh

happy:
	echo ${S3_BUCKET}

sad:
	echo ${AWS_REGION}

train:
	envsubst mnist-training.yaml | kubectl create -f -
