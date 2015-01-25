# =======================================================================================================
# SETUP
# =======================================================================================================
out_w = 1280
out_h = 800
errstep = 1

set terminal unknown
plot 'CollectedRunsResultSerialBIPED.gnuplot.txt' using 1:2 # gather basic statistics
min_biped=GPVAL_DATA_Y_MIN
max_biped=GPVAL_DATA_Y_MAX

set terminal unknown
plot 'CollectedRunsResultSerialQUADRUPED.gnuplot.txt' using 1:2 # gather basic statistics
min_quadruped=GPVAL_DATA_Y_MIN
max_quadruped=GPVAL_DATA_Y_MAX

set terminal unknown
plot 'CollectedRunsResultParallelBIPED2.gnuplot.txt' using 1:2 # gather basic statistics
min_biped2=GPVAL_DATA_Y_MIN
max_biped2=GPVAL_DATA_Y_MAX
set terminal unknown
plot 'CollectedRunsResultParallelBIPED3.gnuplot.txt' using 1:2 # gather basic statistics
min_biped3=GPVAL_DATA_Y_MIN
max_biped3=GPVAL_DATA_Y_MAX
set terminal unknown
plot 'CollectedRunsResultParallelBIPED4.gnuplot.txt' using 1:2 # gather basic statistics
min_biped4=GPVAL_DATA_Y_MIN
max_biped4=GPVAL_DATA_Y_MAX

set terminal unknown
plot 'CollectedRunsResultParallelQUADRUPED2.gnuplot.txt' using 1:2 # gather basic statistics
min_quadruped2=GPVAL_DATA_Y_MIN
max_quadruped2=GPVAL_DATA_Y_MAX
set terminal unknown
plot 'CollectedRunsResultParallelQUADRUPED3.gnuplot.txt' using 1:2 # gather basic statistics
min_quadruped3=GPVAL_DATA_Y_MIN
max_quadruped3=GPVAL_DATA_Y_MAX
plot 'CollectedRunsResultParallelQUADRUPED4.gnuplot.txt' using 1:2 # gather basic statistics
min_quadruped4=GPVAL_DATA_Y_MIN
max_quadruped4=GPVAL_DATA_Y_MAX

#default output
set terminal pngcairo size out_w,out_h enhanced font "Verdana,20"
set output "render/output_raster_ratios_parallel.png"

# settings
set yrange [1:100]
set xrange [1:]
set size .83, 1

set bars small

# define axis
# remove border on top and right and set color to gray
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
# define grid
set style line 12 lc rgb '#AFAFAF' lt 0 lw 1
set grid back ls 12

# color definitions
# lines
set style line 1 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 3 # --- green
set style line 2 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 3 # --- red
set style line 3 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 3 # --- blue
set style line 4 lc rgb '#FF5B00' pt -1 ps 1 lt 1 lw 3 # --- orange
set style line 5 lc rgb '#996633' pt -1 ps 1 lt 1 lw 3 # --- brown
set style line 6 lc rgb '#800080' pt -1 ps 1 lt 1 lw 3 # --- purple
set style line 7 lc rgb '#ADDFAD' pt -1 ps 1 lt 1 lw 3 # --- moss
set style line 8 lc rgb '#000000' pt -1 ps 1 lt 1 lw 3 # --- black
# error bars
set style line 11 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 3 # --- red
set style line 22 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 3 # --- green
set style line 33 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 3 # --- blue

set style fill transparent solid 0.2 noborder

set key off

set xlabel 'Character count'
set ylabel 'Ratio'

set label 2 'Quadruped Parallel 2 ' at 101,48 left tc 		ls 2
set label 3 'Quadruped Parallel 3 ' at 101,43 left tc 		rgb "#C65325"
set label 4 'Biped Parallel 2' at 101,38 left tc 				rgb "#4B632D"
set label 5 'Quadruped Parallel 4' at 101,34 left tc 			ls 6
set label 6 'Biped Parallel 3' at 101,30 left tc 				ls 3
set label 7 'Biped Parallel 4' at 101,23 left tc 					ls 5

set label 8 'Biped Serial ' at 101,74 left tc 		ls 11
set label 9 'Quadruped Serial' at 101,90 left tc 		ls 33


set arrow from 100,37.1237 to 101,43 nohead lc rgb "#C65325"
set arrow from 100,36.1780 to 101,38 nohead lc rgb "#4B632D"

# =======================================================================================================
#
# PLOT ALL
#
# =======================================================================================================
#set autoscale
set ytics 5 font "Calibri,10"
set xtics add ("1" 1)
set xtics 5 font "Calibri,10"
set ytics add ("1" 1)
plot \
"CollectedRunsResultSerialBIPED.gnuplot.txt" using ($1+1):($2/min_biped) with lines ls 11 t 'Biped execution slowdown rate', \
"CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):($2/min_quadruped) with lines ls 33 t 'Quadruped execution slowdown rate', \
"CollectedRunsResultParallelBIPED2.gnuplot.txt" using ($1+1):($2/min_biped2) with lines ls 1 t 'Biped2 execution slowdown rate', \
"CollectedRunsResultParallelQUADRUPED2.gnuplot.txt" using ($1+1):($2/min_quadruped2) with lines ls 2 t 'Quadruped2 execution slowdown rate', \
"CollectedRunsResultParallelBIPED3.gnuplot.txt" using ($1+1):($2/min_biped3) with lines ls 3 t 'Biped3 execution slowdown rate', \
"CollectedRunsResultParallelQUADRUPED3.gnuplot.txt" using ($1+1):($2/min_quadruped3) with lines ls 4 t 'Quadruped3 execution slowdown rate', \
"CollectedRunsResultParallelBIPED4.gnuplot.txt" using ($1+1):($2/min_biped4) with lines ls 5 t 'Biped4 execution slowdown rate', \
"CollectedRunsResultParallelQUADRUPED4.gnuplot.txt" using ($1+1):($2/min_quadruped4) with lines ls 6 t 'Quadruped4 execution slowdown rate', \

# PDF
set terminal pdf enhanced font 'Calibri,10'
set output "render/output_vector_ratios_parallelserial.pdf"
replot


# Live (wxWidgets)
set terminal wxt size out_w,out_h enhanced font 'Verdana,25' persist
replot
