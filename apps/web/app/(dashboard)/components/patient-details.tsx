"use client"

import type { Patient } from "../lib/data"
import { Badge } from "./ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"
import { cn } from "../lib/utils"
import {
  User,
  Phone,
  Mail,
  MapPin,
  Heart,
  Shield,
  AlertTriangle,
  Pill,
  Activity,
  FileText,
  Clock,
  Thermometer,
  Wind,
  Scale,
  Ruler,
} from "lucide-react"

interface PatientDetailsProps {
  patient: Patient
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

export function PatientDetails({ patient }: PatientDetailsProps) {
  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* Header */}
      <div className="mb-6 flex items-start justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/20">
            <span className="text-xl font-semibold text-primary">
              {patient.name
                .split(" ")
                .map((n) => n[0])
                .join("")}
            </span>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground">{patient.name}</h1>
            <p className="text-muted-foreground">
              {patient.id} • {patient.age} years old • {patient.gender}
            </p>
            <div className="mt-2 flex items-center gap-2">
              <Badge className={cn(statusColors[patient.status])}>{patient.status}</Badge>
              <Badge className={cn(priorityColors[patient.priority])}>{patient.priority} Priority</Badge>
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          <button className="rounded-lg bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground hover:bg-secondary/80 transition-colors">
            Edit Patient
          </button>
          <button className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors">
            Start Visit
          </button>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="bg-secondary">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="vitals">Vitals</TabsTrigger>
          <TabsTrigger value="history">Medical History</TabsTrigger>
          <TabsTrigger value="notes">Notes</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Reason for Visit & Symptoms */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-5 w-5 text-primary" />
                Reason for Visit
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-foreground">{patient.reasonForVisit}</p>
              <div>
                <p className="mb-2 text-sm font-medium text-muted-foreground">Current Symptoms</p>
                <div className="flex flex-wrap gap-2">
                  {patient.symptoms.map((symptom, index) => (
                    <Badge
                      key={index}
                      variant="outline"
                      className="bg-destructive/10 text-destructive border-destructive/20"
                    >
                      {symptom}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            {/* Personal Information */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <User className="h-5 w-5 text-primary" />
                  Personal Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-3 text-sm">
                  <Phone className="h-4 w-4 text-muted-foreground" />
                  <span className="text-foreground">{patient.phone}</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  <span className="text-foreground">{patient.email}</span>
                </div>
                <div className="flex items-start gap-3 text-sm">
                  <MapPin className="h-4 w-4 text-muted-foreground mt-0.5" />
                  <span className="text-foreground">{patient.address}</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Heart className="h-4 w-4 text-muted-foreground" />
                  <span className="text-foreground">Blood Type: {patient.bloodType}</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-foreground">DOB: {patient.dateOfBirth}</span>
                </div>
              </CardContent>
            </Card>

            {/* Insurance & Emergency Contact */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Shield className="h-5 w-5 text-primary" />
                  Insurance & Emergency
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Insurance Provider</p>
                  <p className="text-sm text-foreground">{patient.insuranceProvider}</p>
                  <p className="text-xs text-muted-foreground mt-1">{patient.insuranceNumber}</p>
                </div>
                <div className="border-t border-border pt-4">
                  <p className="text-xs text-muted-foreground mb-1">Emergency Contact</p>
                  <p className="text-sm font-medium text-foreground">{patient.emergencyContact.name}</p>
                  <p className="text-xs text-muted-foreground">{patient.emergencyContact.relationship}</p>
                  <p className="text-sm text-foreground mt-1">{patient.emergencyContact.phone}</p>
                </div>
              </CardContent>
            </Card>

            {/* Allergies */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <AlertTriangle className="h-5 w-5 text-warning" />
                  Allergies
                </CardTitle>
              </CardHeader>
              <CardContent>
                {patient.allergies.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {patient.allergies.map((allergy, index) => (
                      <Badge key={index} className="bg-warning/20 text-warning border-0">
                        {allergy}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No known allergies</p>
                )}
              </CardContent>
            </Card>

            {/* Current Medications */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Pill className="h-5 w-5 text-primary" />
                  Current Medications
                </CardTitle>
              </CardHeader>
              <CardContent>
                {patient.currentMedications.length > 0 ? (
                  <ul className="space-y-2">
                    {patient.currentMedications.map((medication, index) => (
                      <li key={index} className="text-sm text-foreground flex items-start gap-2">
                        <span className="h-1.5 w-1.5 rounded-full bg-primary mt-2" />
                        {medication}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No current medications</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="vitals" className="space-y-6">
          <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
            <Card className="bg-card border-border">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-destructive/20">
                    <Activity className="h-5 w-5 text-destructive" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Blood Pressure</p>
                    <p className="text-lg font-semibold text-foreground">{patient.vitalSigns.bloodPressure}</p>
                    <p className="text-xs text-muted-foreground">mmHg</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-4/20">
                    <Heart className="h-5 w-5 text-chart-4" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Heart Rate</p>
                    <p className="text-lg font-semibold text-foreground">{patient.vitalSigns.heartRate}</p>
                    <p className="text-xs text-muted-foreground">bpm</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-warning/20">
                    <Thermometer className="h-5 w-5 text-warning" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Temperature</p>
                    <p className="text-lg font-semibold text-foreground">{patient.vitalSigns.temperature}°</p>
                    <p className="text-xs text-muted-foreground">Fahrenheit</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-info/20">
                    <Wind className="h-5 w-5 text-info" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">O2 Saturation</p>
                    <p className="text-lg font-semibold text-foreground">{patient.vitalSigns.oxygenSaturation}%</p>
                    <p className="text-xs text-muted-foreground">SpO2</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-success/20">
                    <Scale className="h-5 w-5 text-success" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Weight</p>
                    <p className="text-lg font-semibold text-foreground">{patient.vitalSigns.weight}</p>
                    <p className="text-xs text-muted-foreground">lbs</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-3/20">
                    <Ruler className="h-5 w-5 text-chart-3" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Height</p>
                    <p className="text-lg font-semibold text-foreground">{patient.vitalSigns.height}"</p>
                    <p className="text-xs text-muted-foreground">inches</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <FileText className="h-5 w-5 text-primary" />
                Medical History
              </CardTitle>
            </CardHeader>
            <CardContent>
              {patient.medicalHistory.length > 0 ? (
                <ul className="space-y-2">
                  {patient.medicalHistory.map((item, index) => (
                    <li key={index} className="text-sm text-foreground flex items-start gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-primary mt-2" />
                      {item}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">No significant medical history</p>
              )}
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Clock className="h-5 w-5 text-primary" />
                Visit History
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-foreground">
                Last Visit: <span className="font-medium">{patient.lastVisit}</span>
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notes" className="space-y-6">
          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <FileText className="h-5 w-5 text-primary" />
                Clinical Notes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <textarea
                className="w-full h-40 rounded-lg bg-secondary border-0 p-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                placeholder="Add clinical notes for this visit..."
              />
              <div className="mt-4 flex justify-end">
                <button className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors">
                  Save Notes
                </button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
