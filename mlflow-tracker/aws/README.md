# Welcome to your CDK TypeScript project

This is a blank project for CDK development with TypeScript.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Useful commands

* `npm run build`   compile typescript to js
* `npm run watch`   watch for changes and compile
* `npm run test`    perform the jest unit tests
* `cdk deploy`      deploy this stack to your default AWS account/region
* `cdk diff`        compare deployed stack with current state
* `cdk synth`       emits the synthesized CloudFormation template

## Arguments
For `deploy`, use
```
cdk deploy --all --require-approval never
```
for `destroy`, use
```
cdk destroy --all --force
```

## Reference

This repo follows closely with this [tutorial](https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate).
