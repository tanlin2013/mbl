OpenID Connect (OIDC) 
----

When deploying aws cdk on Github Actions,
together with the action [`aws-actions/configure-aws-credentials`](https://github.com/aws-actions/configure-aws-credentials),
one may encounter the error
```
Could not load credentials from any providers
```
this is because OIDC is required to be configured in prior. 

In short, these steps are essential
1. Configure the Identity Provider in the IAM AWS console (use the info in the 2nd link)
2. Create a new role that includes the identity provider <---- important
3. Define the permissions in your YAML file (like below)
4. Add the role's ARN (not the OIDC arn) to the configuration step in the YAML

References
----------
* https://github.com/aws-actions/configure-aws-credentials
* https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services
* https://github.com/aws-actions/configure-aws-credentials/issues/271#issuecomment-1012450577
