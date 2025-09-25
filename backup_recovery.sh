#!/bin/bash
# backup_recovery.sh
# Create backup of recovery results to prevent data loss

BACKUP_DIR="recovery_backup_$(date +%Y%m%d_%H%M%S)"

echo "=== Creating Recovery Backup ==="
echo "Backup directory: $BACKUP_DIR"

if [ ! -d "recovery_out" ]; then
    echo "❌ ERROR: recovery_out directory not found!"
    echo "Nothing to backup."
    exit 1
fi

# Create backup
mkdir -p "$BACKUP_DIR"
cp -r recovery_out/* "$BACKUP_DIR/"

# Verify backup
backup_subjects=$(find "$BACKUP_DIR" -name "recovery_result.csv" | wc -l | xargs)
original_subjects=$(find recovery_out -name "recovery_result.csv" | wc -l | xargs)

echo "✅ Backup created successfully!"
echo "Original subjects: $original_subjects"
echo "Backed up subjects: $backup_subjects"
echo "Backup location: $BACKUP_DIR"

# Create restoration script
cat > "restore_from_${BACKUP_DIR}.sh" << EOF
#!/bin/bash
# Auto-generated restoration script for $BACKUP_DIR

echo "=== Restoring Recovery Data ==="
echo "Restoring from: $BACKUP_DIR"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ ERROR: Backup directory $BACKUP_DIR not found!"
    exit 1
fi

# Remove current recovery_out if it exists
if [ -d "recovery_out" ]; then
    echo "Backing up current recovery_out to recovery_out_old..."
    mv recovery_out recovery_out_old_\$(date +%H%M%S)
fi

# Restore from backup
cp -r "$BACKUP_DIR" recovery_out
echo "✅ Recovery data restored from $BACKUP_DIR"

# Verify restoration
restored_subjects=\$(find recovery_out -name "recovery_result.csv" | wc -l | xargs)
echo "Restored subjects: \$restored_subjects"
EOF

chmod +x "restore_from_${BACKUP_DIR}.sh"
echo "Restoration script created: restore_from_${BACKUP_DIR}.sh"
