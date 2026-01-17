import { generateText, streamText, ollama } from 'ai-sdk-ollama';

// Works in both Node.js and browsers
const { text } = await streamText({
    model: ollama(process.env.OLLAMA_MODEL || 'gemini-3-pro-preview:latest'),
    prompt: userPrompt,
    temperature: 0.8,
});

console.log(text);
