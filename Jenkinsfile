pipeline {
    options {
        timestamps()
        skipDefaultCheckout()
        disableConcurrentBuilds()
    }
    agent {
        node { label 'translator && aws && build' }
    }
    parameters {
        string(name: 'BUILD_VERSION', defaultValue: '', description: 'The build version to deploy (optional)')
        string(name: 'AWS_REGION', defaultValue: 'us-east-1', description: 'AWS Region to deploy')
    }
    triggers {
        pollSCM('H/5 * * * *')
    }
    environment {
        DOCKER_REPO_NAME = "translator-cdskp-openpredict"
        KUBERNETES_BLUE_CLUSTER_NAME = "translator-eks-ci-blue-cluster"
        NAMESPACE = "cdskp"
    }
    stages {
        stage('Checkout source code') {
            steps {
                cleanWs()
                checkout scm
            }
        }
        stage('Build Version'){
            when {expression {return !params.BUILD_VERSION}}
             steps{
                script {
                    BUILD_VERSION_GENERATED = VersionNumber(
                        versionNumberString: 'v${BUILD_YEAR, XX}.${BUILD_MONTH, XX}${BUILD_DAY, XX}.${BUILDS_TODAY}',
                        projectStartDate:    '1970-01-01',
                        skipFailedBuilds:    true)
                    currentBuild.displayName = BUILD_VERSION_GENERATED
                    env.BUILD_VERSION = BUILD_VERSION_GENERATED
                    env.BUILD = 'true'
                }
            }
        }

        stage('Build Docker') {
           when { expression { return env.BUILD == 'true' }}
            steps {
                script {
                    docker.build(env.DOCKER_REPO_NAME, "--no-cache .")
                    docker.withRegistry('https://853771734544.dkr.ecr.us-east-1.amazonaws.com', 'ecr:us-east-1:aws-ifx-deploy') {
                        docker.image(env.DOCKER_REPO_NAME).push("${BUILD_VERSION}")
                    }
                }
            }
        }
        stage('Deploy to AWS EKS Blue') {
            steps {
                sshagent (credentials: ['labshare-svc']) {
                   sh 'git clone git@github.com:Sphinx-Automation/translator-ops.git'
                }
                dir('translator-ops/ops/cdskp/openpredict/') {
                    configFileProvider([
                        configFile(fileId: 'values-ncats.yaml', targetLocation: 'values-ncats.yaml')
                    ]){
                        withAWS(credentials:'aws-ifx-deploy') {
                            sh '''
                            aws --region ${AWS_REGION} eks update-kubeconfig --name ${KUBERNETES_BLUE_CLUSTER_NAME}
                            /bin/bash deploy.sh
                            '''
                        }
                    }
                }
            }
        }
    }
}
