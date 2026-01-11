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
            setLoading(true)
            const data = await loadPatientsFromFilesystem()
            setPatientList(data)
            setLoading(false)
        }
        
        loadPatients()
        
        // Refresh every 10 seconds
        const interval = setInterval(loadPatients, 10000)
        return () => clearInterval(interval)
    }, [])

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
                        <PatientList patients={patientList} selectedPatient={selectedPatient} onSelectPatient={setSelectedPatient} />
                        {selectedPatient ? <PatientDetails patient={selectedPatient} /> : <EmptyState />}
                    </>
                )}
            </main>
        </div>
    )
}
