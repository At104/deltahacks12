"use client"

import { useState } from "react"
import { Sidebar } from "./components/sidebar"
import { PatientList } from "./components/patient-list"
import { PatientDetails } from "./components/patient-details"
import { EmptyState } from "./components/empty-state"
import { patients, type Patient } from "./lib/data"

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("patients")
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex flex-1 overflow-hidden">
        <PatientList patients={patients} selectedPatient={selectedPatient} onSelectPatient={setSelectedPatient} />

        {selectedPatient ? <PatientDetails patient={selectedPatient} /> : <EmptyState />}
      </main>
    </div>
  )
}
