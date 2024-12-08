import React, { useState } from 'react';
import { Button, Alert, Box, CircularProgress } from '@mui/material';

interface ClipboardPasterProps {
    onModelPasted: (model: any) => void;
    buttonText?: string;
    className?: string;
}

const ClipboardPaster: React.FC<ClipboardPasterProps> = ({
    onModelPasted,
    buttonText = "Paste Model",
    className = "",
}) => {
    const [status, setStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
    const [loading, setLoading] = useState<boolean>(false);

    const handlePaste = async () => {
        setStatus(null); // Reset status
        setLoading(true);
        try {
            // Check if the Clipboard API is supported
            if (!navigator.clipboard || !navigator.clipboard.readText) {
                throw new Error("Clipboard API not supported in this browser.");
            }

            // Read text from the clipboard
            const text = await navigator.clipboard.readText();

            if (!text) {
                throw new Error("Clipboard is empty.");
            }

            // Attempt to parse the text as JSON
            let parsedData: any;
            try {
                parsedData = JSON.parse(text);
            } catch (parseError) {
                throw new Error("Clipboard data is not valid JSON.");
            }

            // Call the callback with the parsed data
            onModelPasted(parsedData);
            setStatus({ type: 'success', message: "Model pasted successfully!" });
        } catch (error: any) {
            setStatus({ type: 'error', message: `Error: ${error.message}` });
            console.error("ClipboardPaster Error:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box className={className} display="flex" flexDirection="column" alignItems="center" gap={2}>
            <Button
                variant="contained"
                color="primary"
                onClick={handlePaste}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
            >
                {buttonText}
            </Button>
            {status && (
                <Alert severity={status.type} variant="filled">
                    {status.message}
                </Alert>
            )}
        </Box>
    );
};

export default ClipboardPaster;
