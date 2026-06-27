// A minimal stdio MCP server for integration testing. Implements initialize,
// tools/list (echo), and tools/call (echo echoes {text} back as text content).
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
function send(obj) { process.stdout.write(JSON.stringify(obj) + '\n'); }
rl.on('line', (line) => {
  line = line.trim();
  if (!line) return;
  let req;
  try { req = JSON.parse(line); } catch { return; }
  if (req.id === undefined) return; // notification, no response
  let result;
  switch (req.method) {
    case 'initialize':
      result = { protocolVersion: '2024-11-05', capabilities: { tools: {} }, serverInfo: { name: 'echo-mcp', version: '0.1' } };
      break;
    case 'tools/list':
      result = { tools: [{ name: 'echo', description: 'Echoes text back.', inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] } }] };
      break;
    case 'tools/call':
      const args = (req.params && req.params.arguments) || {};
      result = { content: [{ type: 'text', text: args.text || '' }], isError: false };
      break;
    default:
      send({ jsonrpc: '2.0', id: req.id, error: { code: -32601, message: 'method not found: ' + req.method } });
      return;
  }
  send({ jsonrpc: '2.0', id: req.id, result });
});
