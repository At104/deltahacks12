"use client"

import type { Patient } from "../lib/data"
import { cn } from "../lib/utils"
import { Search } from "lucide-react"
import { Input } from "./ui/input"
import { Badge } from "./ui/badge"
import { useState } from "react"

interface PatientListProps {
  patients: Patient[]
  selectedPatient: Patient | null
  onSelectPatient: (patient: Patient) => void
  isQueueView?: boolean
}

const priorityColors = {
  Low: "bg-info/20 text-info",
  Medium: "bg-warning/20 text-warning",
  High: "bg-chart-4/20 text-chart-4",
  Critical: "bg-destructive/20 text-destructive",
}

const statusColors = {
  Waiting: "bg-muted text-muted-foreground",
  "In Progress": "bg-primary/20 text-primary",
  Completed: "bg-success/20 text-success",
}

export function PatientList({ patients, selectedPatient, onSelectPatient, isQueueView = false }: PatientListProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<"All" | "Waiting" | "In Progress" | "Completed">("All")

  const filteredPatients = patients.filter((patient) => {
    const matchesSearch = 
      patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      patient.id.toLowerCase().includes(searchQuery.toLowerCase())
    
    const matchesStatus = 
      statusFilter === "All" || patient.status === statusFilter
    
    return matchesSearch && matchesStatus
  })

  return (
    <div className="flex h-full w-80 flex-col border-r border-border bg-card">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-card-foreground">
            {isQueueView ? "Triage Queue" : "All Patients"}
          </h2>
          <Badge variant="secondary" className="text-xs">
            {filteredPatients.length} {isQueueView ? "in queue" : "total"}
          </Badge>
        </div>
        {isQueueView && (
          <p className="text-xs text-muted-foreground mb-4">Sorted by priority (highest first)</p>
        )}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search patients..."
            className="pl-9 bg-secondary border-0"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-1 border-b border-border px-4 py-2">
        {(["All", "Waiting", "In Progress", "Completed"] as const).map((status) => (
          <button
            key={status}
            onClick={() => setStatusFilter(status)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              statusFilter === status
                ? "bg-primary/20 text-primary"
                : "text-muted-foreground hover:bg-secondary"
            )}
          >
            {status}
        </button>
        ))}
      </div>

      {/* Patient List */}
      <div className="flex-1 overflow-y-auto">
        {filteredPatients.map((patient) => (
          <button
            key={patient.id}
            onClick={() => onSelectPatient(patient)}
            className={cn(
              "flex w-full flex-col gap-2 border-b border-border p-4 text-left transition-colors hover:bg-secondary/50",
              selectedPatient?.id === patient.id && "bg-secondary",
            )}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                {patient.photo ? (
                  <img
                    src={`http://localhost:3001${patient.photo}`}
                    alt={patient.name}
                    className="h-10 w-10 rounded-full object-cover border-2 border-primary/20"
                  />
                ) : (
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/20">
                  <span className="text-sm font-medium text-primary">
                    {patient.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")}
                  </span>
                </div>
                )}
                <div>
                  <p className="font-medium text-card-foreground">{patient.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {patient.lastVisit}
                  </p>
                </div>
              </div>
              <Badge className={cn("text-xs", priorityColors[patient.priority])}>{patient.priority}</Badge>
            </div>
            <div className="flex items-center justify-between pl-13">
              <p className="text-xs text-muted-foreground line-clamp-1">{patient.reasonForVisit}</p>
              <Badge className={cn("text-xs", statusColors[patient.status])}>{patient.status}</Badge>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
