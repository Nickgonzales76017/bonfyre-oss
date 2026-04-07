# BonfyreTel — FreeSWITCH Telephony Adapter

Pure C binary that connects to FreeSWITCH via Event Socket (plain TCP),
giving Bonfyre full telephony capabilities without any Twilio dependency.

## Architecture

```
Phone Network → SIP Trunk ($1/mo) → FreeSWITCH (MIT, self-hosted)
                                          ↓ Event Socket (TCP :8021)
                                     bonfyre-tel listen
                                          ↓
                           ┌──────────────┼──────────────┐
                     Recording done    SMS received    Call hangup
                           ↓              ↓              ↓
                   bonfyre-pipeline   bonfyre-ingest   SQLite log
                     (async fork)     (async fork)
```

## Quick Start

```bash
# 1. Build
make -C cmd/BonfyreTel

# 2. Install FreeSWITCH (macOS)
brew install freeswitch

# 3. Copy configs
cp cmd/BonfyreTel/deploy/dialplan-bonfyre.xml /usr/local/freeswitch/conf/dialplan/default/
cp cmd/BonfyreTel/deploy/sip-trunk.xml /usr/local/freeswitch/conf/sip_profiles/external/
# Edit sip-trunk.xml with your provider credentials

# 4. Start FreeSWITCH
freeswitch -nonat

# 5. Start listening
./cmd/BonfyreTel/bonfyre-tel listen
```

## Commands

| Command | Description |
|---------|-------------|
| `listen` | Connect to FreeSWITCH ESL, listen for call/SMS events |
| `send-sms` | Send SMS via FreeSWITCH SIP MESSAGE |
| `send-mms` | Send MMS via carrier REST API (curl) |
| `call` | Originate outbound call (optional `--record`) |
| `hangup` | Kill active call by UUID |
| `status` | Show call/message stats from SQLite |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | FreeSWITCH ESL host |
| `--port` | `8021` | FreeSWITCH ESL port |
| `--password` | `ClueCon` | ESL password |
| `--db` | `~/.local/share/bonfyre/tel.db` | SQLite database path |
| `--from` | — | Caller/sender number |
| `--to` | — | Destination number |
| `--body` | — | Message text |
| `--media` | — | MMS media file/URL |
| `--record` | — | Record outbound call |
| `--uuid` | — | Call UUID (for hangup) |

## Event Flow

**Inbound call:**
1. SIP trunk routes call to FreeSWITCH
2. Dialplan answers, records to `.wav`
3. On record completion → ESL event `CHANNEL_EXECUTE_COMPLETE`
4. `bonfyre-tel` catches event → forks `bonfyre-pipeline run <file.wav>`
5. Pipeline: media-prep → transcribe → embed → store

**Inbound SMS:**
1. SIP MESSAGE arrives at FreeSWITCH
2. ESL fires `CUSTOM sms::recv` event
3. `bonfyre-tel` catches event → forks `bonfyre-ingest --text "..."`

**Outbound SMS:**
```bash
bonfyre-tel send-sms --from +15551234567 --to +15559876543 --body "Your transcript is ready"
```

## SIP Trunk Providers

| Provider | Voice $/min | SMS $/msg | DID $/mo | Notes |
|----------|-------------|-----------|----------|-------|
| Telnyx | $0.002 | $0.004 | $1.00 | Best API, SIP + REST |
| Bandwidth | $0.005 | $0.004 | $1.00 | Enterprise-grade |
| VoIP.ms | $0.01 | $0.01 | $0.85 | Budget option |
| SignalWire | $0.01 | $0.01 | $1.00 | FreeSWITCH creators |

## MMS Configuration

MMS requires carrier REST API (SIP doesn't support MMS natively):

```bash
export BONFYRE_TEL_MMS_ENDPOINT="https://api.telnyx.com/v2/messages"
export BONFYRE_TEL_API_KEY="your-api-key"

bonfyre-tel send-mms --from +15551234567 --to +15559876543 \
    --body "See attached" --media /path/to/file.pdf
```

## Cost Comparison

| | Twilio | BonfyreTel + Telnyx |
|---|--------|---------------------|
| Voice | $0.013/min | $0.002/min (85% less) |
| SMS | $0.0079/msg | $0.004/msg (49% less) |
| DID | $1.15/mo | $1.00/mo |
| MMS | $0.02/msg | $0.01/msg (50% less) |
| Vendor lock-in | Total | Zero |
| Code ownership | None | 100% |
