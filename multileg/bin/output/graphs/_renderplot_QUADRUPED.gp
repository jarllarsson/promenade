# =======================================================================================================
# SETUP
# =======================================================================================================
out_w = 1280
out_h = 800
errstep = 1
#default output
set terminal pngcairo size out_w,out_h enhanced font "Calibri,18"
set output "render/output_raster_frames_QUADRUPED.png"

# settings
set yrange [0:0.3]
set xrange [0:799]
# set autoscale ymax

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
set style line 1 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 3 # --- green
set style line 2 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 3 # --- red
set style line 3 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 3 # --- blue
set style line 4 lc rgb '#FF5B00' pt -1 ps 1 lt 1 lw 3 # --- orange (for fit)
# error bars
set style line 11 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 0.3 # --- red
set style line 22 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 0.3 # --- green
set style line 33 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 0.3 # --- blue

set style fill transparent solid 0.7 noborder

set key off

set xlabel 'Step'
set ylabel 'Milliseconds'

set label 2 'Quadruped Serial' at 100,0.26 left tc rgb "#00541A"
set label 3 'Quadruped Parallel (2 invocations)' at 100,0.06 left tc rgb "#550200"

# =======================================================================================================
#
# PLOT ALL
#
# =======================================================================================================
#set autoscale
set ytics 0.05 font "Calibri,10"
set xtics 100 font "Calibri,10"
set xtics add ("799" 798)
plot \
"perf_serial20QUADRUPED.gnuplot.txt" using 1:4:5 with filledcurves lc rgb "#88A61B", \
										"" using 1:2 with lines lc rgb "#4F873C" lw 1.5 t 'Quadruped Serial', \
"perf_parallel20QUADRUPED_thread2.gnuplot.txt" using 1:4:5 with filledcurves lc rgb "#7F75A8", \
										"" using 1:2 with lines lc rgb "#972F52" lw 1.5 t 'Quadruped Parallel (2)'

# PDF
set terminal pdf enhanced font 'Calibri,15'
set output "render/output_vector_frames_QUADRUPED.pdf"
replot


# Live (wxWidgets)
set terminal wxt size out_w,out_h enhanced font 'Verdana,25' persist
replot
