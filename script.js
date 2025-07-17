class FashionChatbot {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.apiUrl = 'http://localhost:8000';
        this.selectedSku = null;
        this.lastItems = [];
        
        this.initializeElements();
        this.attachEventListeners();
    }

    generateSessionId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    initializeElements() {
        this.chatbotButton = document.getElementById('chatbot-button');
        this.chatbotContainer = document.getElementById('chatbot-container');
        this.closeChatbot = document.getElementById('close-chatbot');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.voiceButton = document.getElementById('voice-button');
        this.languageSelect = document.getElementById('language-select');
        // Remove thumbnail container references as we won't need them
    }

    attachEventListeners() {
        this.chatbotButton.addEventListener('click', () => this.toggleChatbot());
        this.closeChatbot.addEventListener('click', () => this.closeChatbotWidget());
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.chatInput.addEventListener('input', () => this.updateSendButton());
    }

    toggleChatbot() {
        this.chatbotContainer.classList.toggle('hidden');
        if (!this.chatbotContainer.classList.contains('hidden')) {
            this.chatInput.focus();
        }
    }

    closeChatbotWidget() {
        this.chatbotContainer.classList.add('hidden');
    }

    updateSendButton() {
        const hasText = this.chatInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText;
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        // Check for ordinal references if no SKU is selected
        if (!this.selectedSku) {
            const ordinalSku = this.resolveOrdinalReference(message);
            if (ordinalSku) {
                this.selectedSku = ordinalSku;
            }
        }

        this.addMessage('user', message);
        this.chatInput.value = '';
        this.updateSendButton();

        const loadingId = this.addLoadingMessage();

        try {
            const response = await this.callChatAPI(message);
            this.removeLoadingMessage(loadingId);
            this.handleChatResponse(response);
        } catch (error) {
            this.removeLoadingMessage(loadingId);
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            console.error('Chat API error:', error);
        }
    }

    async callChatAPI(message) {
        const payload = {
            session_id: this.sessionId,
            message: message,
            language: this.languageSelect.value,
            selected_sku: this.selectedSku
        };

        const response = await fetch(`${this.apiUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    // handleChatResponse(response) {
    //     // Add the assistant's text message
    //     this.addMessage('assistant', response.assistant_reply);
        
    //     // If there are items, add them as a product grid message
    //     if (response.last_items && response.last_items.length > 0) {
    //         this.lastItems = response.last_items;
    //         this.addProductsMessage(response.last_items);
    //     }

    //     this.selectedSku = response.selected_sku || null;
    //     this.scrollToBottom();
    // }
    handleChatResponse(response) {
    this.addMessage('assistant', response.assistant_reply);

    if (response.last_items && response.last_items.length > 0) {
        const newSkus = response.last_items.map(it => it.sku);
        const oldSkus = this.lastItems.map(it => it.sku);
        const changed =
            newSkus.length !== oldSkus.length ||
            newSkus.some((s, i) => s !== oldSkus[i]);

        if (changed) {
            this.lastItems = response.last_items;
            this.addProductsMessage(response.last_items);
        }
    }

    this.selectedSku = response.selected_sku || null;
    this.scrollToBottom();
}

    addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);
        
        this.scrollToBottom();
        return messageDiv;
    }

    addProductsMessage(items) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant products-message';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const productsCarousel = document.createElement('div');
        productsCarousel.className = 'products-carousel';
        
        const productsTrack = document.createElement('div');
        productsTrack.className = 'products-track';
        
        items.forEach((item, index) => {
            const productCard = document.createElement('div');
            productCard.className = 'product-card';
            productCard.setAttribute('data-sku', item.sku);
            
            const imageUrl = item.image_url || item.url || 'https://via.placeholder.com/150x150';
            
            productCard.innerHTML = `
                <div class="product-image">
                    <img src="${imageUrl}" alt="Product ${index + 1}">
                </div>
                <div class="product-info">
                    <div class="product-brand">${item.brand || 'Brand'}</div>
                    <div class="product-name">${item.name || 'Product ' + (index + 1)}</div>
                    <div class="product-price">AED ${item.sale_price || item.price || 'N/A'}</div>
                </div>
            `;
            
            productCard.addEventListener('click', () => {
                this.selectProduct(item.sku, productCard);
            });
            
            productsTrack.appendChild(productCard);
        });
        
        productsCarousel.appendChild(productsTrack);
        messageContent.appendChild(productsCarousel);
        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);
        
        this.scrollToBottom();
    }

    selectProduct(sku, cardElement) {
        this.selectedSku = sku;
        
        // Remove previous selections
        const allCards = this.chatMessages.querySelectorAll('.product-card');
        allCards.forEach(card => {
            card.classList.remove('selected');
        });
        
        // Add selection to clicked card
        cardElement.classList.add('selected');
        
        // Add a follow-up message
        this.addMessage('assistant', 'Great choice! I can help you with this item - product details, styling tips, care instructions, or anything else you\'d like to know. Or I can even help you add this to cart.');
    }

    addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant loading';
        loadingDiv.innerHTML = `
            <div class="message-content">
                <div class="loading">
                    Thinking...
                    <div class="loading-dots">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(loadingDiv);
        this.scrollToBottom();
        return loadingDiv;
    }

    removeLoadingMessage(loadingElement) {
        if (loadingElement && loadingElement.parentNode) {
            loadingElement.parentNode.removeChild(loadingElement);
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    resolveOrdinalReference(message) {
        const ordinalWords = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
        };
        
        const numericMatch = message.match(/\b(\d+)(?:st|nd|rd|th)?\b/);
        if (numericMatch) {
            const index = parseInt(numericMatch[1]) - 1;
            if (index >= 0 && index < this.lastItems.length) {
                return this.lastItems[index].sku;
            }
        }
        
        for (const [word, number] of Object.entries(ordinalWords)) {
            if (message.toLowerCase().includes(word)) {
                const index = number - 1;
                if (index >= 0 && index < this.lastItems.length) {
                    return this.lastItems[index].sku;
                }
            }
        }
        
        return null;
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', () => {
    new FashionChatbot();
});

