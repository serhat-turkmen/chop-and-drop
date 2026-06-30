# Mounting an external drive

Recordings often live on an external SSD/HDD. This guide shows how to make that
drive available so you can point **Chop & Drop** at it (via **Add folder…**).

- **macOS / Windows:** external drives mount automatically when plugged in. On
  macOS they appear under `/Volumes/<NAME>`; on Windows as a drive letter
  (`E:\`, `F:\`, …). No manual step needed; skip to the bottom.
- **Linux (Ubuntu/GNOME):** USB drives usually auto-mount too, but sometimes you
  need to mount them by hand. The rest of this guide covers that.

---

## 1. Check whether it's already mounted

Most of the time a plugged-in drive is already available. List the disks:

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,LABEL
```

Example output:

```
NAME        SIZE TYPE FSTYPE MOUNTPOINT                 LABEL
sdb         3.6T disk
└─sdb1      3.6T part exfat  /media/serhat/DAISY_SSD    DAISY_SSD
```

If your drive shows a **MOUNTPOINT** (e.g. `/media/<you>/DAISY_SSD`), it's ready;
that path is the folder to use in Chop & Drop. On Ubuntu, auto-mounted drives live
under `/media/<your-username>/<LABEL>`.

## 2. Mount it manually (if there's no mountpoint)

Identify the partition (the `partN` line, e.g. `sdc1`) and its filesystem from the
`lsblk` output, then:

```bash
sudo mkdir -p /mnt/ext
sudo mount /dev/sdc1 /mnt/ext        # replace sdc1 with your partition
```

The filesystem type is usually auto-detected. Your files are now under `/mnt/ext`.

### Mount as yourself (so you can write without sudo)

`exFAT`/`NTFS`/`FAT` drives don't carry Linux permissions, so mount them with your
user/group to get full read-write access:

```bash
id -u; id -g                          # note your uid and gid (often 1000 / 1000)
sudo mount -o uid=$(id -u),gid=$(id -g) /dev/sdc1 /mnt/ext
```

## 3. Filesystem drivers (one-time)

Most distros include these already; install only if a mount fails with
"unknown filesystem type":

```bash
sudo apt update
sudo apt install exfat-fuse exfatprogs ntfs-3g
```

| Drive formatted as | Driver package |
|--------------------|----------------|
| exFAT (common cross-platform) | `exfat-fuse`, `exfatprogs` |
| NTFS (Windows)                | `ntfs-3g` |
| ext4 (Linux)                  | built in |

## 4. Unmount safely when done

Always unmount before unplugging, or you risk corrupting files:

```bash
sudo umount /mnt/ext                  # or the auto-mount path, e.g. /media/<you>/LABEL
```

In the GNOME Files app you can also click the ⏏ (eject) icon next to the drive.

---

## Troubleshooting

**The drive shows `0 B` and no partitions** (e.g. `sda  0B disk` in `lsblk`)
This is not a mount problem; the drive isn't initializing. Try:

1. Unplug and replug, preferably into a **rear USB port** and a different cable
   (front ports / hubs often under-power 3.5" or spinning drives).
2. Watch the kernel log as you plug it in:
   ```bash
   sudo dmesg -w
   ```
   (insert the drive, read the new lines, `Ctrl-C` to stop). Errors here point to a
   power, cable, or hardware fault.
3. Re-list devices to see if a size/partition now appears:
   ```bash
   lsblk -o NAME,SIZE,FSTYPE,LABEL
   ```

**"mount: only root can do that"**: prefix the command with `sudo`.

**"wrong fs type, bad option, bad superblock"**: usually a missing driver
(see step 3) or a corrupted/unformatted partition.

**Permission denied writing to the drive**: remount with the `uid=/gid=` options
shown in step 2 (exFAT/NTFS), or `sudo chown -R $(id -u):$(id -g) /mnt/ext` (ext4).

---

## Using the drive in Chop & Drop

Once mounted, in the app click **Add folder…** and navigate to the mount path:

- Linux: `/media/<you>/<LABEL>` (auto-mount) or `/mnt/ext` (manual)
- macOS: `/Volumes/<NAME>`
- Windows: the drive letter, e.g. `E:\`

With **Include subfolders** ticked, Chop & Drop recursively finds every video on
the drive and processes them one by one. Tip: leave the **Output folder** blank to
write results next to each source video on the same drive, or set it to a folder on
your fast internal disk for quicker writes.
