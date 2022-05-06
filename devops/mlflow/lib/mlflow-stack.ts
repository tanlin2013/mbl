import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sm from 'aws-cdk-lib/aws-secretsmanager';
import { VPCStack } from './vpc-stack';
import { DBStack } from './db-stack';
import { FargateStack } from './fargate-stack';


export class MlflowStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // IAM Role
    const servicePrincipal = new iam.ServicePrincipal('ecs-tasks.amazonaws.com');
    const role = new iam.Role(this, 'TASKROLE', {
      assumedBy: servicePrincipal
    })
    role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonECS_FullAccess'))

    // Secret
    const dbSecret = new sm.Secret(this, 'DBSECRET', {
      secretName: 'dbPassword',
      generateSecretString: {
        passwordLength: 20,
        excludePunctuation: true
      }
    })

    // VPC
    const vpcStack = new VPCStack(this, 'VPCStack');

    // Database
    const dbStack = new DBStack(this, 'DBStack', {
      vpc: vpcStack.mlflowVPC,
      dbSecret: dbSecret
    });

    // Fargate Service
    new FargateStack(this, 'FargateStack', {
      vpc: vpcStack.mlflowVPC,
      role: role,
      dbEndpointAddr: dbStack.MYSQL.clusterEndpoint.socketAddress,
      dbSecret: dbSecret
    });

  }
}
