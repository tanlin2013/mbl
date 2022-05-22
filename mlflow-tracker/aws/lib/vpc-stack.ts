import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';


export class VPCStack extends Stack {
    public readonly mlflowVPC: ec2.Vpc

    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);

        const publicSubnet: ec2.SubnetConfiguration = {
            name: 'Public',
            subnetType: ec2.SubnetType.PUBLIC,
            cidrMask: 28
        };
        const privateSubnet: ec2.SubnetConfiguration = {
            name: 'Private',
            subnetType: ec2.SubnetType.PRIVATE_WITH_NAT,
            cidrMask: 28
        };
        const isolatedSubnet: ec2.SubnetConfiguration = {
            name: 'DB',
            subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
            cidrMask: 28
        };

        const vpc = new ec2.Vpc(this, 'VPC', {
            cidr: '10.0.0.0/24',
            maxAzs: 2,
            natGatewayProvider: ec2.NatProvider.gateway(),
            natGateways: 1,
            subnetConfiguration: [publicSubnet, privateSubnet, isolatedSubnet]
        })

        this.mlflowVPC = vpc;
    }
}
