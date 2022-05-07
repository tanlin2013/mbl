import { RemovalPolicy, Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as sm from 'aws-cdk-lib/aws-secretsmanager';


interface DBProps extends StackProps {
    vpc: ec2.Vpc;
    dbSecret: sm.Secret;
}


export class DBStack extends Stack {
    public readonly MYSQL: rds.ServerlessCluster

    constructor(scope: Construct, id: string, props: DBProps) {
        super(scope, id, props);

        const sgRDS = new ec2.SecurityGroup(this, 'SGRDS', {
            securityGroupName: 'sg_rds',
            vpc: props.vpc
        })
        sgRDS.addIngressRule(
            ec2.Peer.ipv4('10.0.0.0/24'),
            ec2.Port.tcp(3306)
        )

        const database = new rds.ServerlessCluster(this, 'MYSQL', {
            engine: rds.DatabaseClusterEngine.AURORA_MYSQL,
            defaultDatabaseName: 'MlflowDB',
            vpc: props.vpc,
            vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
            securityGroups: [sgRDS],
            credentials: rds.Credentials.fromUsername('master', { password: props.dbSecret.secretValue }),
            removalPolicy: RemovalPolicy.DESTROY,
            deletionProtection: false
        })

        this.MYSQL = database;

    }
}
