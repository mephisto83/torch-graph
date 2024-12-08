import React, { useRef } from 'react';
import { Box, Button, Stack } from '@mui/material';

interface GraphControlsProps {
    handleSaveGraph: () => void;
    handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

const GraphControls: React.FC<GraphControlsProps> = ({
    handleSaveGraph,
    handleFileChange,
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    return (
        <Box mt={2}>
            <Stack direction="row" spacing={2}>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSaveGraph}
                >
                    Save Graph
                </Button>
                <Button
                    variant="contained"
                    color="secondary"
                    onClick={() => fileInputRef.current?.click()}
                >
                    Load Graph
                </Button>
                <input
                    type="file"
                    accept=".json"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                />
            </Stack>
        </Box>
    );
};

export default GraphControls;
