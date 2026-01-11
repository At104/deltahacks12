"use client";

import { useState, useEffect } from "react";
import CameraFeed from "./components/CameraFeed";
import Transcription from "./components/Transcription";
import styles from "./page.module.css";
import { PatientRecord, loadPatientRecords, savePatientRecords } from "./lib/patientStorage";

export default function Home() {
  const [step, setStep] = useState<"name" | "choice" | "input" | "camera">("name");
  const [patientName, setPatientName] = useState("");
  const [nameInputMethod, setNameInputMethod] = useState<"text" | "voice" | null>(null);
  const [inputMethod, setInputMethod] = useState<"text" | "voice" | null>(null);
  const [symptoms, setSymptoms] = useState("");
  const [textInput, setTextInput] = useState("");
  const [patientRecords, setPatientRecords] = useState<PatientRecord[]>([]);
  const [currentRecordId, setCurrentRecordId] = useState<string | null>(null);
  const [photoData, setPhotoData] = useState<string | null>(null);
  const [language, setLanguage] = useState<string>("en");
  const [isSaving, setIsSaving] = useState(false);

  // Load records from localStorage on mount
  useEffect(() => {
    setPatientRecords(loadPatientRecords());
  }, []);

  // Save to localStorage whenever records change
  useEffect(() => {
    if (patientRecords.length > 0) {
      savePatientRecords(patientRecords);
    }
  }, [patientRecords]);

  const handleNameMethodChoice = (method: "text" | "voice") => {
    setNameInputMethod(method);
  };

  const handleNameSubmit = (name: string) => {
    // Extract just the name from voice input (e.g., "Hello, my name is John" -> "John")
    let extractedName = name;
    
    // Common patterns for name introduction
    const patterns = [
      /(?:my name is|i am|i'm|this is|call me)\s+(.+?)(?:\.|$)/i,
      /(?:name|called):\s*(.+?)(?:\.|$)/i,
    ];
    
    for (const pattern of patterns) {
      const match = name.match(pattern);
      if (match && match[1]) {
        extractedName = match[1].trim();
        break;
      }
    }
    
    // Clean up any remaining punctuation
    extractedName = extractedName.replace(/[.,!?]$/, '').trim();
    
    setPatientName(extractedName);
    setStep("choice");
  };

  const handleTextNameSubmit = () => {
    if (patientName.trim()) {
      setStep("choice");
    }
  };

  const handleMethodChoice = (method: "text" | "voice") => {
    setInputMethod(method);
    setStep("input");
  };

  const handleSymptomsSubmit = (symptomText: string, detectedLanguage?: string) => {
    setSymptoms(symptomText);
    setLanguage(detectedLanguage || "en");
    setStep("camera");
    
    // Create a new patient record
    const newRecord: PatientRecord = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      symptoms: symptomText,
      inputMethod: inputMethod!,
      language: detectedLanguage,
    };
    
    setCurrentRecordId(newRecord.id);
    setPatientRecords(prev => [...prev, newRecord]);
    
    console.log("✅ Patient record created:", newRecord.id);
  };

  const handleTextSubmit = () => {
    if (textInput.trim()) {
      handleSymptomsSubmit(textInput, "en");
    }
  };

  const handlePhotoCapture = async (imageData: string) => {
    setPhotoData(imageData);
    
    // Update the current record with the photo
    if (currentRecordId) {
      setPatientRecords(prev => 
        prev.map(record => 
          record.id === currentRecordId 
            ? { ...record, photo: imageData }
            : record
        )
      );
      console.log("📸 Photo captured");
    }

    // Save to local filesystem
    await saveToFileSystem(imageData);
  };

  const saveToFileSystem = async (photo: string) => {
    setIsSaving(true);
    try {
      const response = await fetch('http://localhost:3001/save-patient', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: patientName,
          symptoms,
          photo,
          language,
          inputMethod,
          timestamp: new Date().toISOString(),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save patient record');
      }

      const data = await response.json();
      console.log('✅ Saved to filesystem:', data.folder);
      alert(`Patient record saved successfully!\nFolder: ${data.folder}`);
    } catch (error) {
      console.error('❌ Error saving to filesystem:', error);
      alert('Error saving to filesystem. Make sure the file storage server is running.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setStep("name");
    setPatientName("");
    setNameInputMethod(null);
    setInputMethod(null);
    setSymptoms("");
    setTextInput("");
    setCurrentRecordId(null);
    setPhotoData(null);
    setLanguage("en");
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Hospital Triage System</h1>
        <p className={styles.subtitle}>Patient Check-In</p>
        {patientRecords.length > 0 && (
          <p className={styles.recordCount}>
            {patientRecords.length} patient{patientRecords.length !== 1 ? 's' : ''} recorded
          </p>
        )}
      </header>
      <main className={styles.main}>
        {step === "name" && (
          <div className={styles.choiceContainer}>
            <h2>What is your name?</h2>
            {nameInputMethod === null && (
              <div className={styles.buttonGroup}>
                <button
                  className={styles.choiceButton}
                  onClick={() => handleNameMethodChoice("text")}
                >
                  <span className={styles.icon}>⌨️</span>
                  Type Name
                </button>
                <button
                  className={styles.choiceButton}
                  onClick={() => handleNameMethodChoice("voice")}
                >
                  <span className={styles.icon}>🎤</span>
                  Say Name
                </button>
              </div>
            )}
            {nameInputMethod === "text" && (
              <div className={styles.inputSection}>
                <input
                  type="text"
                  className={styles.nameInput}
                  placeholder="Enter your name"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  autoFocus
                />
                <button
                  className={styles.nextButton}
                  onClick={handleTextNameSubmit}
                  disabled={!patientName.trim()}
                >
                  Next
                </button>
              </div>
            )}
            {nameInputMethod === "voice" && (
              <div className={styles.inputSection}>
                <Transcription onTranscriptionComplete={handleNameSubmit} />
              </div>
            )}
          </div>
        )}

        {step === "choice" && (
          <div className={styles.choiceContainer}>
            <h2>Hello, {patientName}!</h2>
            <p className={styles.questionText}>How would you like to describe your symptoms?</p>
            <div className={styles.buttonGroup}>
              <button
                className={styles.choiceButton}
                onClick={() => handleMethodChoice("text")}
              >
                <span className={styles.icon}>⌨️</span>
                Type Symptoms
              </button>
              <button
                className={styles.choiceButton}
                onClick={() => handleMethodChoice("voice")}
              >
                <span className={styles.icon}>🎤</span>
                Voice Record
              </button>
            </div>
            <button className={styles.backButton} onClick={() => setStep("name")}>
              Back
            </button>
          </div>
        )}

        {step === "input" && inputMethod === "text" && (
          <div className={styles.inputContainer}>
            <h2>Type your symptoms</h2>
            <textarea
              className={styles.textArea}
              placeholder="Describe your symptoms here..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              rows={6}
            />
            <div className={styles.buttonGroup}>
              <button className={styles.backButton} onClick={() => setStep("choice")}>
                Back
              </button>
              <button
                className={styles.nextButton}
                onClick={handleTextSubmit}
                disabled={!textInput.trim()}
              >
                Next: Take Photo
              </button>
            </div>
          </div>
        )}

        {step === "input" && inputMethod === "voice" && (
          <div className={styles.inputContainer}>
            <h2>Record your symptoms</h2>
            <Transcription onTranscriptionComplete={(text) => handleSymptomsSubmit(text)} />
            <button className={styles.backButton} onClick={() => setStep("choice")}>
              Back
            </button>
          </div>
        )}

        {step === "camera" && (
          <div className={styles.cameraContainer}>
            <h2>Take a photo for assessment</h2>
            <p className={styles.symptomsDisplay}>
              <strong>Patient:</strong> {patientName}<br />
              <strong>Symptoms:</strong> {symptoms}
            </p>
            <CameraFeed onCapture={handlePhotoCapture} />
            {isSaving && (
              <div className={styles.savingIndicator}>
                <div className={styles.spinner}></div>
                Saving to filesystem...
              </div>
            )}
            <button className={styles.backButton} onClick={handleReset}>
              Start Over
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
