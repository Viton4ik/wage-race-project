# Data sources and method

All values are GROSS pay in EUR per month, July 2000 to July 2025, for the 22 EU countries that have a
statutory minimum wage (AT, DK, FI, IT, SE have none and are not shown).

| Series | Source | Notes |
|---|---|---|
| Minimum wage | Eurostat `earn_mw_cur` | Value on 1 Jan and 1 Jul each year, 12-month basis. Joined with monotone (PCHIP) interpolation. A country enters the race at its first value: Ireland Jul 2000, Croatia Jul 2008, Germany Jan 2015, Cyprus Jan 2023. |
| Average wage | Eurostat `nama_10_fte`, average full-time adjusted salary per employee | Annual / 12, placed at mid-year, PCHIP between years. 2025 = 2024 x growth of `earn_nt_net` (gross, average worker) 2024-2025. |
| Average wage, Netherlands | Eurostat `earn_nt_net` (gross, average worker) | NL has no `nama_10_fte` series. 2024 = 2023 x euro-area `nama_10_fte` growth [estimate], because `earn_nt_net` NL shows a methodological break in 2024. |
| Bulgaria 2000, Poland 2000 average wage | backcast with the 2001-2002 growth rate | [estimate] |

Raw download: `raw/eurostat_wages.txt` (Eurostat API, 2026-10-31). Rebuild: `python3 scripts/build_data.py`.

## Known limitations
* Lithuania 2019: gross wages (and the minimum wage) were scaled up ~29 % by a tax reform that moved social
  contributions from employers to employees. The jump is real in gross terms, not a pay rise.
* Nominal euros: non-euro currencies (BG, CZ, HU, PL, RO) move with the exchange rate.
* The "%" label is the minimum wage as a share of the average wage (a Kaitz-type index), not of the median.
* The animation uses a 7-month centred average; the first and the last month are exact.
