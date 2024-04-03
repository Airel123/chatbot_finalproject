class Chatbox {
    constructor() {
        this.args = {
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            closeButton: document.querySelector('.close-btn')
        }
        this.state = false;
        this.messages = [];
    }

    display() {
        const {chatBox, sendButton, closeButton} = this.args;
        sendButton.addEventListener('click', () => this.onSendButton(chatBox))
        closeButton.addEventListener('click', () => this.closeChat());
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        // if empty
        if (text1 === "") {
            return;
        }

        // create user messages array
        let msg1 = {name: "User", message: text1}
        this.messages.push(msg1);

        // Send to server
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({message: text1}),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                // create bot messages array.
                let msg2 = {name: "Bot", message: r.answer};
                this.messages.push(msg2);
                // update the interface
                this.updateChatText(chatbox)
                // Clear the input field
                textField.value = ''

            }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
        });
    }

    updateChatText(chatbox) {
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = '';  // 清空当前消息

        this.messages.forEach((item) => {
            const messageElement = document.createElement('div');
            messageElement.classList.add('messages__item');
            messageElement.classList.add(item.name === "Bot" ? 'messages__item--visitor' : 'messages__item--operator');
            messageElement.textContent = item.message;
            chatmessage.appendChild(messageElement);
        });

        // 延迟滚动到底部以确保DOM已更新
        setTimeout(() => {
            chatmessage.scrollTop = chatmessage.scrollHeight;
        }, 0);
    }

    // display a thankyou message.
    closeChat() {
        // removing original child elements
        while (this.args.chatBox.firstChild) {
            this.args.chatBox.removeChild(this.args.chatBox.firstChild);
        }

        this.args.chatBox.classList.add('chatbox-centered');

        const thankYouMessage = document.createElement('div');
        thankYouMessage.className = 'chatbox__thankyou-message';
        thankYouMessage.textContent = 'Thanks for using our chat service!';
        this.args.chatBox.appendChild(thankYouMessage);
    }
}
const chatbox = new Chatbox();
chatbox.display();