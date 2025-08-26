const ApiBuilder = require('claudia-api-builder');
const { OpenAI } = require('openai');
const AWS = require('aws-sdk');
const jwt = require('jsonwebtoken');

const api = new ApiBuilder();
module.exports = api;

// Initialize AWS SDK for Secrets Manager and Cognito
const secretsManager = new AWS.SecretsManager();
const cognito = new AWS.CognitoIdentityServiceProvider();

// Micro-Grok inspired processing function
const processAlchemistRequest = async (inputPayload, context) => {
  try {
    // 1. OAuth2.0 Token Validation
    const authHeader = inputPayload.headers.Authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      throw new Error('Missing or invalid Authorization header');
    }
    const token = authHeader.split(' ')[1];
    const decoded = await verifyOAuthToken(token);

    // 2. Sanitize and Translate Input
    const sanitizedInput = await sanitizeInput(inputPayload.body.userInput);
    const translatedPrompt = `
      [System: You are the Alchemist Orchestrator with micro-Grok reasoning. Translate and enrich the input for Vial agent training and communication.]
      User Input: "${sanitizedInput}"
      Context: ${JSON.stringify(inputPayload.body.context || {})}
      Protocol: MAML-compliant JSON
    `;

    // 3. Micro-Grok Reasoning (using lightweight LLM)
    const openai = new OpenAI({
      apiKey: await getSecret('OPENAI_API_KEY'),
    });
    const chatCompletion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are a lightweight reasoning agent inspired by xAI Grok. Clarify intents, orchestrate training tasks, and ensure MAML compliance.' },
        { role: 'user', content: translatedPrompt }
      ],
      temperature: 0.1
    });

    const orchestratedOutput = chatCompletion.choices[0].message.content;

    // 4. Reputation Validation
    const reputationValid = await validateReputation(decoded.walletAddress, inputPayload.body.reputation);
    if (!reputationValid) {
      throw new Error('Insufficient reputation score for operation');
    }

    // 5. Return Structured Response
    return {
      status: 'success',
      originalInput: inputPayload.body,
      orchestratedOutput: JSON.parse(orchestratedOutput),
      timestamp: new Date().toISOString(),
      alchemistContext: { user: decoded.sub, wallet: decoded.walletAddress }
    };
  } catch (error) {
    console.error('Alchemist Agent Error:', error);
    throw new Error(`Orchestration failed: ${error.message}`);
  }
};

// API Endpoint
api.post('/alchemist', async (request) => {
  const { userInput, context, reputation } = request.body;
  if (!userInput || !reputation) {
    return new api.ApiResponse(
      { error: 'userInput and reputation are required' },
      { 'Content-Type': 'application/json' },
      400
    );
  }
  try {
    const result = await processAlchemistRequest(request, 'HTTP Request');
    return result;
  } catch (error) {
    return new api.ApiResponse(
      { error: error.message },
      { 'Content-Type': 'application/json' },
      500
    );
  }
}, { success: 201 });

// Helper Functions
async function verifyOAuthToken(token) {
  try {
    const params = { AccessToken: token };
    const user = await cognito.getUser(params).promise();
    return {
      sub: user.UserAttributes.find(attr => attr.Name === 'sub').Value,
      walletAddress: user.UserAttributes.find(attr => attr.Name === 'custom:walletAddress').Value
    };
  } catch (error) {
    throw new Error('OAuth token verification failed');
  }
}

async function getSecret(secretName) {
  const data = await secretsManager.getSecretValue({ SecretId: secretName }).promise();
  return data.SecretString;
}

async function sanitizeInput(input) {
  // Call external Python service for sanitization
  const response = await fetch('http://localhost:8000/api/services/sanitize', {
    method: 'POST',
    body: JSON.stringify({ input }),
    headers: { 'Content-Type': 'application/json' }
  });
  const result = await response.json();
  return result.sanitizedInput;
}

async function validateReputation(walletAddress, reputation) {
  // Call external Python service for reputation validation
  const response = await fetch('http://localhost:8000/api/services/validate_reputation', {
    method: 'POST',
    body: JSON.stringify({ walletAddress, reputation }),
    headers: { 'Content-Type': 'application/json' }
  });
  const result = await response.json();
  return result.isValid;
}

// Deployment Instructions
// Path: webxos-vial-mcp/src/hermes/claudia/alchemist_agent.js
// Run: npm install claudia-api-builder openai aws-sdk jsonwebtoken
// Deploy: claudia create --name alchemist-agent --region us-east-1 --api-module alchemist_agent
// Update: claudia update --name alchemist-agent
