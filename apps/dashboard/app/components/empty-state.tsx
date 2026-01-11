import { Users } from "lucide-react"

export function EmptyState() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center p-6">
      <div className="flex h-20 w-20 items-center justify-center rounded-full bg-secondary mb-4">
        <Users className="h-10 w-10 text-muted-foreground" />
      </div>
      <h2 className="text-xl font-semibold text-foreground mb-2">Select a Patient</h2>
      <p className="text-muted-foreground text-center max-w-md">
        Choose a patient from the list to view their personal information, symptoms, and visit details.
      </p>
    </div>
  )
}
