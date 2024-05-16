import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Button from '@mui/material/Button';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import Box from '@mui/material/Box';

const ResultsTable = ({ results, onResetSearch }) => {
    return (
        <div>
            <Table size="small" aria-label="a dense table">
                <TableHead>
                    <TableRow>
                        <TableCell sx={{ color: '#ABADC6', fontWeight: 'bold' }}>Index</TableCell>
                        <TableCell sx={{ color: '#ABADC6', fontWeight: 'bold' }}>Table Name</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {results.map((row, index) => (
                        <TableRow key={index}>
                            <TableCell component="th" scope="row" sx={{ color: '#ABADC6' }}>
                                {index + 1}
                            </TableCell>
                            <TableCell sx={{ color: '#ABADC6' }}>{row.table_name}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Button
                    variant="outlined"
                    size="small"
                    startIcon={<RestartAltIcon />}
                    onClick={onResetSearch}
                    sx={{
                        margin: '15px 0px',
                        fontSize: '0.8rem',
                    }}
                >
                    Reset Search Space from Here
                </Button>
            </Box>
        </div>
    );
};

export default ResultsTable;
