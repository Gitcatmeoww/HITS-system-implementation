import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MessageList from './MessageList';
import InputArea from './InputArea';
import { Box } from '@mui/material';
import CircularProgress from '@mui/material/CircularProgress';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import ResultsTable from './ResultsTable';
import Chip from '@mui/material/Chip';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';

function ChatContainer() {
    const [messages, setMessages] = useState([]);
    const [isInitialMessage, setIsInitialMessage] = useState(true);
    const [threadId, setThreadId] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');
    const [currentSearchSpace, setCurrentSearchSpace] = useState([]);
    const [isChipExpanded, setIsChipExpanded] = useState(true);


    useEffect(() => {
        startNewThread();
    }, []);

    const startNewThread = async () => {
        setIsLoading(true);

        try {
            const response = await axios.post('http://127.0.0.1:5000/api/start_chat');
            const { thread_id } = response.data;
            sessionStorage.setItem('threadId', thread_id);
            setThreadId(thread_id);
        } catch (error) {
            console.error('Error starting a new thread:', error);
            setError('Could not start a new chat thread. Please try again.');
        }

        setIsLoading(false);
    };

    const sendMessage = async (messageText) => {
        if (!threadId) {
            setError('There is no active chat thread. Please start a new chat.');
            return;
        }

        const chatHistoryUrl = 'http://127.0.0.1:5000/api/update_chat_history'

        const searchUrl = isInitialMessage
            ? 'http://127.0.0.1:5000/api/hyse_search'
            : 'http://127.0.0.1:5000/api/refine_search_space';

        setIsLoading(true);

        try {
            // Update the chat history
            const chatHistoryResponse = await axios.post(chatHistoryUrl, {
                thread_id: threadId,
                query: messageText,
            });

            console.log(chatHistoryResponse)

            // Check if chat history update was successful before proceeding
            if (chatHistoryResponse.data.success) {
                // Perform the hyse search or refinement based on the initial message status
                const searchResponse = await axios.post(searchUrl, {
                    thread_id: threadId,
                    query: messageText,
                });

                const newMessage = { id: messages.length + 1, text: messageText, sender: 'user' };

                let additionalInfo = '';
                if (!isInitialMessage) {
                    additionalInfo = (
                        <>
                            <p>Inferred Action: {searchResponse.data.inferred_action}</p>
                            <p>Mentioned Semantic Fields: {searchResponse.data.mention_semantic_fields.join(', ')}</p>
                            <p>Mentioned Raw Fields: {searchResponse.data.mention_raw_fields.join(', ')}</p>
                        </>
                    );
                }

                const completeResults = searchResponse.data.complete_results;

                // Update the current search space
                setCurrentSearchSpace(completeResults);

                const reply = {
                    id: messages.length + 2,
                    text: (
                        <>
                            {additionalInfo}
                            {searchResponse.data.top_results && searchResponse.data.top_results.length > 0
                                ? <ResultsTable
                                    results={searchResponse.data.top_results}
                                    onResetSearch={() => handleResetSearch(completeResults)} />
                                : <p>No results found or an error occurred.</p>}
                        </>
                    ),
                    sender: 'system',
                };

                setMessages(prevMessages => [...prevMessages, newMessage, reply]);
                setIsInitialMessage(false);
            } else {
                // Handle failure to update chat history
                console.error('Failed to update chat history:', chatHistoryResponse.data.error);
                setError('Could not update chat history. Please try again.');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            setError('Could not send the message. Please try again.');
        }

        setIsLoading(false);
    };

    const handleResetSearch = async (completeResults) => {
        try {
            const response = await axios.post('http://127.0.0.1:5000/api/reset_search_space', {
                thread_id: threadId,
                results: completeResults,
            });

            // Set success message
            setSuccessMessage('Search space has been successfully reset.');
            // Update the current search space
            setCurrentSearchSpace(completeResults);
        } catch (error) {
            console.error('Error resetting search space:', error);
            setError('Could not reset search space. Please try again.');
        }
    };

    const handleCloseSnackbar = () => {
        setError('');
        setSuccessMessage('');
    };

    const toggleChipExpansion = () => {
        setIsChipExpanded(!isChipExpanded);
    };

    return (
        <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            maxWidth: '80%',
            height: '100vh',
            margin: 'auto',
            paddingBottom: '56px'
        }}>
            <Box sx={{
                width: '100%',
                overflow: 'auto',
                flexGrow: 1,
                marginBottom: '8px'
            }}>
                <MessageList messages={messages} />
                {isLoading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                        <CircularProgress />
                    </Box>
                )}
            </Box>
            <InputArea onSendMessage={sendMessage} />

            {currentSearchSpace.length > 0 && (
                <Chip
                    icon={isChipExpanded ? <KeyboardArrowRightIcon /> : <KeyboardArrowLeftIcon />}
                    label={isChipExpanded ? `Current Search Space: ${currentSearchSpace.length} tables` : ''}
                    sx={{ position: 'fixed', bottom: 100, right: 30, fontSize: '0.75rem', fontWeight: 'bold', color: '#ABADC6', backgroundColor: '#424867' }}
                    onClick={toggleChipExpansion}
                />
            )}

            <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseSnackbar}>
                <Alert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
                    {error}
                </Alert>
            </Snackbar>
            <Snackbar
                open={!!successMessage}
                autoHideDuration={3000}
                onClose={() => setSuccessMessage('')}
                anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
                <Alert onClose={() => setSuccessMessage('')} severity="success" sx={{ width: '100%' }}>
                    {successMessage}
                </Alert>
            </Snackbar>
        </Box>
    );
}

export default ChatContainer;
