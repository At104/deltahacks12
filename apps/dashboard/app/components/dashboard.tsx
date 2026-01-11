"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "./sidebar"
import { PatientList } from "./patient-list"
import { PatientDetails } from "./patient-details"
import { EmptyState } from "./empty-state"
import { patients, loadPatientsFromFilesystem, type Patient } from "../lib/data"

export default function Dashboard() {
    const [activeTab, setActiveTab] = useState("patients")
    const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)
    const [patientList, setPatientList] = useState<Patient[]>(patients)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadPatients() {
            const data = await loadPatientsFromFilesystem()
            
            // Smart merge: keep existing patients, add new ones
            setPatientList(prev => {
                if (prev.length === 0) return data
                
                // Create a map of existing patients by ID
                const existingMap = new Map(prev.map(p => [p.id, p]))
                
                // Merge: keep existing patient data, only add NEW patients
                return data.map(newPatient => {
                    const existing = existingMap.get(newPatient.id)
                    if (existing) {
                        // Keep existing patient's status/priority (user may have changed them)
                        // But update other fields like photo path
                        return {
                            ...newPatient,
                            status: existing.status,
                            priority: existing.priority
                        }
                    }
                    return newPatient
                })
            })
            
            setLoading(false)
        }
        
        loadPatients()
        
        // Refresh every 10 seconds (but smart merge so no visual flash)
        const interval = setInterval(loadPatients, 10000)
        return () => clearInterval(interval)
    }, [])

    const updatePatientStatus = async (patientId: string, newStatus: Patient["status"]) => {
        // Optimistically update UI
        setPatientList(prev => 
            prev.map(p => p.id === patientId ? { ...p, status: newStatus } : p)
        )
        
        if (selectedPatient?.id === patientId) {
            setSelectedPatient({ ...selectedPatient, status: newStatus })
        }

        // Persist to backend
        try {
            await fetch('http://localhost:3001/update-patient', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder: patientId, status: newStatus }),
            })
        } catch (error) {
            console.error('Failed to update status:', error)
        }
    }

    const updatePatientPriority = async (patientId: string, newPriority: Patient["priority"]) => {
        // Optimistically update UI
        setPatientList(prev => 
            prev.map(p => p.id === patientId ? { ...p, priority: newPriority } : p)
        )
        
        if (selectedPatient?.id === patientId) {
            setSelectedPatient({ ...selectedPatient, priority: newPriority })
        }

        // Persist to backend
        try {
            await fetch('http://localhost:3001/update-patient', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder: patientId, priority: newPriority }),
            })
        } catch (error) {
            console.error('Failed to update priority:', error)
        }
    }

    // Get queue-sorted patients
    const queuePatients = [...patientList]
        .filter(p => p.status !== "Completed")
        .sort((a, b) => {
            const priorityOrder = { Critical: 0, High: 1, Medium: 2, Low: 3 }
            return priorityOrder[a.priority] - priorityOrder[b.priority]
        })

    return (
        <div className="flex h-screen overflow-hidden bg-background">
            <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

            <main className="flex flex-1 overflow-hidden">
                {loading ? (
                    <div className="flex items-center justify-center flex-1">
                        <p className="text-muted-foreground">Loading patients...</p>
                    </div>
                ) : (
                    <>
                        <PatientList 
                            patients={activeTab === "queue" ? queuePatients : patientList}
                            selectedPatient={selectedPatient}
                            onSelectPatient={setSelectedPatient}
                            isQueueView={activeTab === "queue"}
                        />
                        {selectedPatient ? (
                            <PatientDetails 
                                patient={selectedPatient}
                                onUpdateStatus={updatePatientStatus}
                                onUpdatePriority={updatePatientPriority}
                            />
                        ) : (
                            <EmptyState />
                        )}
                    </>
                )}
            </main>
        </div>
    )
}
