# =======================================================================================================
# SETUP
# =======================================================================================================
out_w = 1280
out_h = 800
errstep = 1
#default output
set terminal pngcairo size out_w,out_h enhanced font "Verdana,20"
set output "render/collectedRuns_QUADRUPED_raster.png"

# settings
set yrange [0:1.2]
set xrange [1:]
set size .94, 1

set bars small

# define axis
# remove border on top and right and set color to gray
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
# define grid
set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12

# color definitions
# lines
set style line 1 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 2 # --- green
set style line 2 lc rgb '#48D6B0' pt -1 ps 1 lt 1 lw 2 # --- turqoise
set style line 3 lc rgb '#8C55B0' pt -1 ps 1 lt 1 lw 2 # --- purple
set style line 4 lc rgb '#FF5B00' pt -1 ps 1 lt 1 lw 2 # --- orange
# error bars
set style line 11 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 0.3 # --- red
set style line 22 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 0.3 # --- green
set style line 33 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 0.3 # --- blue

set style fill transparent solid 0.2 noborder

set key off

set xlabel 'Quadruped Characters'
set ylabel 'Milliseconds'

set label 2 'Serial' at 101,0.99 left tc rgb "#196D33"
set label 3 'Parallel 2' at 101,0.57 left tc rgb "#306A70"
set label 4 'Parallel 3' at 101,0.47 left tc rgb "#38196B"
set label 5 'Parallel 4' at 101,0.4 left tc rgb "#CD4C18"

# =======================================================================================================
#
# PLOT ALL
#
# =======================================================================================================
#set autoscale
set ytics 0.05 font "Calibri,10"
set xtics 5 font "Calibri,10"
set xtics add ("1" 1)
plot \
"CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):4:5 with filledcurves lc rgb "#88A61B" title 'serial error', \
														 "" using ($1+1):2 with lines ls 1 t 'Serial', \
"CollectedRunsResultParallelQUADRUPED2.gnuplot.txt" using ($1+1):4:5 with filledcurves lc rgb "#61D891" title 'parallel2 error', \
															"" using ($1+1):2 with lines ls 2 t 'Parallel2', \
"CollectedRunsResultParallelQUADRUPED3.gnuplot.txt" using ($1+1):4:5 with filledcurves lc rgb "#D555B0" title 'parallel3 error', \
															"" using ($1+1):2 with lines ls 3 t 'Parallel3', \
"CollectedRunsResultParallelQUADRUPED4.gnuplot.txt" using ($1+1):4:5 with filledcurves lc rgb "#FF4D26" title 'parallel4 error', \
															"" using ($1+1):2 with lines ls 4 t 'Parallel4'


# PDF
set terminal pdf enhanced font 'Calibri,10'
set output "render/collectedRuns_QUADRUPED_vector.pdf"
replot


# Live (wxWidgets)
set terminal wxt size out_w,out_h enhanced font 'Verdana,25' persist
replot

