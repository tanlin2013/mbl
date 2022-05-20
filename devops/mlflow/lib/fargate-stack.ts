import { CfnOutput, Duration, Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecsPatterns from 'aws-cdk-lib/aws-ecs-patterns';
import * as sm from 'aws-cdk-lib/aws-secretsmanager';
import * as path from 'path';
import { StringParameter } from 'aws-cdk-lib/aws-ssm';


declare const secret: sm.Secret;


interface FargateProps extends StackProps {
    vpc: ec2.Vpc;
    role: iam.Role;
    dbEndpointAddr: string;
    dbSecret: sm.Secret;
}


export class FargateStack extends Stack {
    constructor(scope: Construct, id: string, props: FargateProps) {
        super(scope, id, props);

        const cluster = new ecs.Cluster(this, 'CLUSTER', {
            clusterName: 'mlflow',
            vpc: props.vpc
        })

        const taskDefinition = new ecs.FargateTaskDefinition(this, 'Mlflow', {
            taskRole: props.role
        })

        const container = taskDefinition.addContainer('Container', {
            image: ecs.ContainerImage.fromAsset(path.join(__dirname, '..')),
            environment: {
                'HOST': props.dbEndpointAddr,
                'DATABASE': 'MlflowDB',
                'USERNAME': 'master',
                'ARTIFACTPATH': 'data/mlflow',
            },
            secrets: {
                'PASSWORD': ecs.Secret.fromSecretsManager(props.dbSecret),
                'SFTPUSER': ecs.Secret.fromSsmParameter(
                    StringParameter.fromSecureStringParameterAttributes(this, 'SFTPUSER', {
                        parameterName: '/mlflow/sftp/user'
                    })
                ),
                'SFTPHOST': ecs.Secret.fromSsmParameter(
                    StringParameter.fromSecureStringParameterAttributes(this, 'SFTPHOST', {
                        parameterName: '/mlflow/sftp/host'
                    })
                ),
            },
            logging: ecs.LogDriver.awsLogs({ streamPrefix: 'mlflow' })
        })

        container.addPortMappings({
            containerPort: 5000,
            hostPort: 5000,
            protocol: ecs.Protocol.TCP
        })

        const fargateService = new ecsPatterns.NetworkLoadBalancedFargateService(this, 'MLFLOW', {
            serviceName: 'mlflow',
            cluster,
            taskDefinition
        })

        const securityGroup = fargateService.service.connections.securityGroups[0].addIngressRule(
            ec2.Peer.ipv4(props.vpc.vpcCidrBlock),
            ec2.Port.tcp(5000),
            'Allow inbound from VPC for mlflow'
        )

        const scaling = fargateService.service.autoScaleTaskCount({
            maxCapacity: 2
        })
        scaling.scaleOnCpuUtilization('AUTOSCALING', {
            targetUtilizationPercent: 70,
            scaleInCooldown: Duration.seconds(60),
            scaleOutCooldown: Duration.seconds(60)
        })

        new CfnOutput(this, 'LoadBalancerDNS', { value: fargateService.loadBalancer.loadBalancerDnsName });

    }
}
