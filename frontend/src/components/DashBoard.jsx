import { useState, useEffect } from 'react';

function Dashboard() {
    const [chats, setChats] = useState([]);
    const [selectedChat, setSelectedChat] = useState(null);

    useEffect(() => {
        fetch('/api/list_chats/', { credentials: 'include' }) // ✅ Fixed URL
            .then(res => res.json())
            .then(data => setChats(data.sessions)); // ✅ sessions not chats
    }, []);

    const loadChat = (sessionId) => {
        fetch(`/api/get_chat/${sessionId}/`, { credentials: 'include' }) // ✅ Fixed URL
            .then(res => res.json())
            .then(data => setSelectedChat(data.messages));
    };

    return (
        <div style={{ display: 'flex' }}>
            {/* Sidebar */}
            <div style={{ width: '30%', borderRight: '1px solid gray', padding: '10px' }}>
                <h2>My Chats</h2>
                <button onClick={() => setSelectedChat(null)}>➕ New Chat</button>
                {chats.map(chat => (
                    <div key={chat._id} onClick={() => loadChat(chat._id)} style={{ cursor: 'pointer', marginTop: '10px' }}>
                        {chat._id || 'No Title'}
                    </div>
                ))}
            </div>

            {/* Chat Window */}
            <div style={{ flex: 1, padding: '10px' }}>
                {selectedChat ? (
                    <div>
                        <h2>Chat</h2>
                        {selectedChat.map((msg, index) => (
                            <div key={index} style={{ margin: '5px 0' }}>
                                <b>{msg.role === 'user' ? 'You' : 'Bot'}:</b> {msg.prompt || msg.response}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div>Start a new chat!</div>
                )}
            </div>
        </div>
    );
}

export default Dashboard;
