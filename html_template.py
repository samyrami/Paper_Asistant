css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #f8f9fa;
}

.chat-message.user {
    background-color: #e9ecef;
    border-left: 5px solid #1e3d59;
}

.chat-message.assistant {
    background-color: #f8f9fa;
    border-left: 5px solid #17a2b8;
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 68px;
    max-height: 68px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #212529;
}

/* Estilos para las sugerencias */
.stExpander {
    border-left: 5px solid #28a745;
    background-color: #f8f9fa;
    margin-top: 1rem;
}

/* Estilo para el input del chat */
.stTextInput > div > div {
    border-radius: 20px;
}

/* Estilo para los botones */
.stButton > button {
    border-radius: 20px;
    width: 100%;
}
</style>
'''
