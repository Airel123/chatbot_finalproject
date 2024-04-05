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
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        closeButton.addEventListener('click', () => this.closeChat());
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });

        this.showOpenLines(); // 显示机器人的开场白
    }

    showOpenLines() {
        // 定义开场白消息内容
        let openLinesMsg = {name: "Bot", message: "Hello there! 👋 I'm your Chatbot, ready to talk anytime you need. "};
        this.messages.push(openLinesMsg); // 将开场白消息添加到消息数组
        this.updateChatText(this.args.chatBox); // 立即更新聊天界面以显示消息
    }

    onSendButton(chatbox) {
    var textField = chatbox.querySelector('input');
    let text1 = textField.value;
    if (text1 === "") {
        return;
    }

    let msg1 = {name: "User", message: text1};
    this.messages.push(msg1);
    this.updateChatText(chatbox);
    textField.value = '';

    let thinkingMsg = {name: "Bot", message: ".", id: "thinking"};
    this.messages.push(thinkingMsg);
    this.updateChatText(chatbox);

    let dots = 1;
    const intervalId = setInterval(() => {
        dots = dots % 3 + 1; // Cycle through 1 to 3
        thinkingMsg.message = ".".repeat(dots);
        this.updateChatText(chatbox); // Update the message with new dots
    }, 100); // Change dots every 500 milliseconds

    // setTimeout(() => {
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
            clearInterval(intervalId); // Stop the dots animation
            this.messages = this.messages.filter(m => m.id !== "thinking"); // Remove the thinking message
            let msg2 = {name: "Bot", message: r.answer};
            this.messages.push(msg2);
            this.updateChatText(chatbox);
        }).catch((error) => {
            console.error('Error:', error);
            clearInterval(intervalId);
            this.messages = this.messages.filter(m => m.id !== "thinking");
            this.updateChatText(chatbox);
        });}
    // }, 0);




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
        thankYouMessage.innerHTML = 'Thank you for sharing your time <br> with me today.(●\'◡\'●)';
        // thankYouMessage.textContent = 'Thank you for sharing your time \n with me today.(●\'◡\'●) ';
        this.args.chatBox.appendChild(thankYouMessage);
    }
}
const chatbox = new Chatbox();
chatbox.display();