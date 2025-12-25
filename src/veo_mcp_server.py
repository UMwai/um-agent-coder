import sys
import json
import os
import requests
import logging

# Configure logging to stderr so it doesn't interfere with MCP on stdout
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='[VeoMCP] %(message)s')
logger = logging.getLogger()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "veo-3.1-generate-preview-001"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

def generate_video(prompt: str, fps: int = 24) -> str:
    """
    Generates a video using Google's Veo 3.1 model.
    """
    if not API_KEY:
        return "Error: GOOGLE_API_KEY environment variable is not set."

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    generate_url = f"{API_BASE_URL}/{MODEL_NAME}:predict"
    
    payload = {
        "instances": [
            {
                "prompt": prompt,
                "video_parameters": {
                    "fps": fps
                }
            }
        ]
    }

    try:
        logger.info(f"Sending request to Veo 3.1 for prompt: {prompt}")
        response = requests.post(generate_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return f"Video generation successful. Result: {json.dumps(result)}"
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return f"Error generating video: {str(e)}"

def handle_message(message):
    msg_type = message.get("method")
    msg_id = message.get("id")
    
    if msg_type == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "veo-video-generator",
                    "version": "1.0.0"
                }
            }
        }
    
    elif msg_type == "notifications/initialized":
        # No response needed for notifications
        return None

    elif msg_type == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "generate_video",
                        "description": "Generates a video using Google's Veo 3.1 model based on a text prompt.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The text description of the video to generate."
                                },
                                "fps": {
                                    "type": "integer",
                                    "description": "Frames per second (default: 24)",
                                    "default": 24
                                }
                            },
                            "required": ["prompt"]
                        }
                    }
                ]
            }
        }

    elif msg_type == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "generate_video":
            result_text = generate_video(args.get("prompt"), args.get("fps", 24))
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ],
                    "isError": result_text.startswith("Error")
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            }
            
    elif msg_type == "ping":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {}
        }

    return None

def main():
    # Use unbuffered stdin/stdout
    sys.stdin = os.fdopen(sys.stdin.fileno(), 'rb', buffering=0)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', buffering=0)
    
    buffer = b""
    
    while True:
        try:
            # Read functionality roughly implementing JSON-RPC message framing if needed, 
            # but MCP usually just sends one JSON object per line for stdio transport 
            # OR uses Content-Length header.
            # Let's implement the Content-Length header reading which is standard for MCP.
            
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.decode('utf-8').strip()
            if not line:
                continue
                
            if line.startswith("Content-Length:"):
                length = int(line.split(":")[1].strip())
                # Skip empty line
                sys.stdin.readline()
                # Read body
                body = sys.stdin.read(length)
                if not body:
                    break
                
                message = json.loads(body.decode('utf-8'))
                response = handle_message(message)
                
                if response:
                    response_json = json.dumps(response)
                    # Write with header
                    sys.stdout.write(f"Content-Length: {len(response_json)}\r\n\r\n{response_json}".encode('utf-8'))
                    sys.stdout.flush()
            else:
                # Fallback for simple line-based JSON (sometimes used in testing)
                try:
                    message = json.loads(line)
                    response = handle_message(message)
                    if response:
                         # Send back as simple line if we received as simple line? 
                         # Ideally stick to header format for response.
                        response_json = json.dumps(response)
                        sys.stdout.write(f"Content-Length: {len(response_json)}\r\n\r\n{response_json}".encode('utf-8'))
                        sys.stdout.flush()
                except:
                    pass

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            # Don't crash the server, just log
            pass

if __name__ == "__main__":
    main()