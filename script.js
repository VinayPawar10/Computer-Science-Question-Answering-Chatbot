document.addEventListener("DOMContentLoaded", function() {
    const chatbox = document.getElementById("chatbox");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const statusIndicator = document.getElementById("status-indicator");

    // Set initial status
    statusIndicator.classList.add("active");

    // Generate or retrieve session ID
    const sessionId = generateSessionId();

    // Add welcome message
    setTimeout(() => {
        displayMessage("Hello! I'm your CS assistant. How can I help you today?", "bot-message");
    }, 500);

    // Event listeners
    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === "") return;

        displayMessage(message, "user-message");
        userInput.value = "";
        
        // Show typing indicator
        const typingIndicator = createTypingIndicator();
        chatbox.appendChild(typingIndicator);
        chatbox.scrollTop = chatbox.scrollHeight;

        try {
            const response = await fetch("/get-response", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: sessionId
                })
            });

            // Remove typing indicator
            chatbox.removeChild(typingIndicator);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayMessage(data.reply, "bot-message");
            
            // Log entities for debugging (can be used for UI enhancements)
            if (data.entities && Object.keys(data.entities).length > 0) {
                console.log("Detected entities:", data.entities);
            }
        } catch (error) {
            console.error("Error:", error);
            displayMessage("Sorry, I'm having trouble responding right now. Please try again later.", "bot-message");
        }
    }

    function displayMessage(text, className) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", className);
        messageElement.textContent = text;
        chatbox.appendChild(messageElement);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function createTypingIndicator() {
        const typingElement = document.createElement("div");
        typingElement.classList.add("message", "bot-message");
        typingElement.id = "typing-indicator";
        
        const typingText = document.createElement("div");
        typingText.classList.add("typing-indicator");
        typingText.innerHTML = '<span></span><span></span><span></span>';
        
        typingElement.appendChild(typingText);
        return typingElement;
    }

    function generateSessionId() {
        // Generate or retrieve a session ID
        if (!localStorage.getItem('chatbot_session_id')) {
            const newId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
            localStorage.setItem('chatbot_session_id', newId);
        }
        return localStorage.getItem('chatbot_session_id');
    }

    // Focus input on load
    userInput.focus();
});
