// Package main is the HTTP gateway entry point.
package main

// Import standard library packages used by the gateway.
import (
	// bytes helps create an io.Reader from the request body for proxying
	"bytes"
	// encoding/json validates and inspects JSON payloads
	"encoding/json"
	// io is used to read and copy HTTP bodies
	"io"
	// log prints startup and proxy error messages
	"log"
	// net/http provides the HTTP server and client
	"net/http"
	// time configures the upstream request timeout
	"time"
)

// aiServiceURL is the upstream FastAPI chat endpoint.
const aiServiceURL = "http://localhost:8000/chat"

// client is a shared HTTP client with a hard timeout for upstream calls.
var client = &http.Client{Timeout: 90 * time.Second}

// chatHandler proxies validated /chat requests to the AI service.
func chatHandler(w http.ResponseWriter, r *http.Request) {
	// Allow only POST because chat requires a JSON body.
	if r.Method != http.MethodPost {
		// Return a clear method error to the caller.
		http.Error(w, `{"error":"POST only"}`, http.StatusMethodNotAllowed)
		// Stop handling this request.
		return
	}

	// Read the full request body into memory.
	body, err := io.ReadAll(r.Body)
	// Handle body read failures.
	if err != nil {
		// Tell the client the body could not be read.
		http.Error(w, `{"error":"cannot read body"}`, http.StatusBadRequest)
		// Stop handling this request.
		return
	}

	// Temporary map used only for lightweight JSON validation.
	var payload map[string]interface{}
	// Ensure the body is valid JSON object content.
	if err := json.Unmarshal(body, &payload); err != nil {
		// Reject malformed JSON early.
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		// Stop handling this request.
		return
	}
	// Require the AI service contract field: message.
	if _, ok := payload["message"]; !ok {
		// Reject payloads that cannot be served by /chat.
		http.Error(w, `{"error":"missing 'message' key"}`, http.StatusBadRequest)
		// Stop handling this request.
		return
	}

	// Forward the original JSON body to the AI service.
	resp, err := client.Post(aiServiceURL, "application/json", bytes.NewReader(body))
	// Handle upstream connectivity / timeout errors.
	if err != nil {
		// Log the detailed internal error for operators.
		log.Printf("AI service error: %v", err)
		// Return a generic bad-gateway response to the client.
		http.Error(w, `{"error":"AI service unavailable"}`, http.StatusBadGateway)
		// Stop handling this request.
		return
	}
	// Always close the upstream response body.
	defer resp.Body.Close()

	// Tell the client we are returning JSON.
	w.Header().Set("Content-Type", "application/json")
	// Preserve the upstream status code.
	w.WriteHeader(resp.StatusCode)
	// Copy the upstream body bytes to the client unchanged.
	io.Copy(w, resp.Body)
}

// main starts the gateway HTTP server.
func main() {
	// Register the chat proxy route.
	http.HandleFunc("/chat", chatHandler)
	// Announce the listen address in logs.
	log.Println("Gateway listening on :8080")
	// Block forever serving HTTP on port 8080.
	log.Fatal(http.ListenAndServe(":8080", nil))
}
