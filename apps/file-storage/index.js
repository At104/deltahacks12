import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Directory for patient records
const RECORDS_DIR = path.join(__dirname, '../../patient_records');

// Serve patient photos statically
app.use('/patient-photos', express.static(RECORDS_DIR));

// Ensure records directory exists
await fs.mkdir(RECORDS_DIR, { recursive: true });

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'file-storage' });
});

app.post('/save-patient', async (req, res) => {
  try {
    const { name, symptoms, photo, language, inputMethod, timestamp } = req.body;

    if (!name || !symptoms) {
      return res.status(400).json({ error: 'Name and symptoms are required' });
    }

    // Sanitize patient name for folder name
    const sanitizedName = name.replace(/[^a-z0-9_-]/gi, '_').toLowerCase();
    const timestamp_str = new Date(timestamp || Date.now()).toISOString().replace(/[:.]/g, '-');
    const folderName = `${sanitizedName}_${timestamp_str}`;
    const patientDir = path.join(RECORDS_DIR, folderName);

    // Create patient directory
    await fs.mkdir(patientDir, { recursive: true });

    // Save transcript
    const transcriptPath = path.join(patientDir, 'transcript.txt');
    const transcriptContent = `Patient Name: ${name}
Timestamp: ${new Date(timestamp || Date.now()).toISOString()}
Input Method: ${inputMethod || 'unknown'}
Language: ${language || 'unknown'}

SYMPTOMS:
${symptoms}
`;
    await fs.writeFile(transcriptPath, transcriptContent, 'utf-8');

    // Save photo if provided
    if (photo) {
      const photoPath = path.join(patientDir, 'photo.jpg');
      // Remove data URL prefix if present
      const base64Data = photo.replace(/^data:image\/\w+;base64,/, '');
      await fs.writeFile(photoPath, base64Data, 'base64');
    }

    // Save metadata as JSON
    const metadataPath = path.join(patientDir, 'metadata.json');
    const metadata = {
      name,
      symptoms,
      language,
      inputMethod,
      timestamp: timestamp || new Date().toISOString(),
      hasPhoto: !!photo,
    };
    await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2), 'utf-8');

    console.log(`✅ Saved patient record: ${folderName}`);

    res.json({
      success: true,
      folder: folderName,
      path: patientDir,
    });
  } catch (error) {
    console.error('❌ Error saving patient:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/list-patients', async (req, res) => {
  try {
    const folders = await fs.readdir(RECORDS_DIR);
    const patients = [];

    for (const folder of folders) {
      const metadataPath = path.join(RECORDS_DIR, folder, 'metadata.json');
      const photoPath = path.join(RECORDS_DIR, folder, 'photo.jpg');
      
      try {
        const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf-8'));
        
        // Check if photo exists
        let hasPhoto = false;
        try {
          await fs.access(photoPath);
          hasPhoto = true;
        } catch (e) {
          // Photo doesn't exist
        }
        
        patients.push({ 
          folder, 
          ...metadata,
          photoPath: hasPhoto ? `/patient-photos/${folder}/photo.jpg` : null,
          status: metadata.status || 'Waiting',
          priority: metadata.priority || 'Medium'
        });
      } catch (e) {
        // Skip folders without metadata
      }
    }

    res.json({ patients });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/update-patient', async (req, res) => {
  try {
    const { folder, status, priority } = req.body;

    if (!folder) {
      return res.status(400).json({ error: 'Folder name is required' });
    }

    const metadataPath = path.join(RECORDS_DIR, folder, 'metadata.json');
    
    // Read existing metadata
    const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf-8'));
    
    // Update fields
    if (status !== undefined) {
      metadata.status = status;
    }
    if (priority !== undefined) {
      metadata.priority = priority;
    }
    
    // Write back
    await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2), 'utf-8');

    console.log(`✅ Updated patient ${folder}: status=${status}, priority=${priority}`);

    res.json({ success: true, metadata });
  } catch (error) {
    console.error('❌ Error updating patient:', error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = 3001;
app.listen(PORT, () => {
  console.log(`\n📁 File Storage API running on http://localhost:${PORT}`);
  console.log(`📂 Saving patient records to: ${RECORDS_DIR}\n`);
});
