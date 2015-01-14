# =======================================================================================================
# SETUP
# =======================================================================================================
out_w = 1280
out_h = 800
errstep = 1
#default output
set terminal pngcairo size out_w,out_h enhanced font "Verdana,20"
set output "render/collectedRuns_ALL_raster.png"

# settings
set yrange [0:1.2]
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
set style line 11 lc rgb '#8b9946' pt -1 ps 1 lt 1 lw 0.3 # --- red
set style line 22 lc rgb '#5c1720' pt -1 ps 1 lt 1 lw 0.3 # --- green
set style line 33 lc rgb '#76a1c8' pt -1 ps 1 lt 1 lw 0.3 # --- blue

set style fill transparent solid 0.2 noborder

set key off

set xlabel 'Character Count'
set ylabel 'Milliseconds'

set label 2 'Quadruped Serial' at 101,0.99 left tc 		rgb "#4B632D"
set label 3 'Quadruped Parallel 2' at 101,0.57 left tc 	ls 2
set label 4 'Quadruped Parallel 3' at 101,0.47 left tc 	rgb "#4088C4"
set label 5 'Quadruped Parallel 4' at 101,0.4 left tc 	rgb "#C65325"
set label 6 'Biped Serial' at 101,0.34 left tc 				ls 5
set label 7 'Biped Parallel 2' at 101,0.2 left tc 			ls 6
set label 8 'Biped Parallel 3' at 101,0.16 left tc 			rgb "#658979"
set label 9 'Biped Parallel 4' at 101,0.12 left tc 			ls 8
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
"CollectedRunsResultSerialQUADRUPED.gnuplot.txt" using ($1+1):2 with lines ls 1 t 'Quadruped Serial', \
"CollectedRunsResultParallelQUADRUPED2.gnuplot.txt" using ($1+1):2 with lines ls 2 t 'Quadruped Parallel2', \
"CollectedRunsResultParallelQUADRUPED3.gnuplot.txt" using ($1+1):2 with lines ls 3 t 'Quadruped Parallel3', \
"CollectedRunsResultParallelQUADRUPED4.gnuplot.txt" using ($1+1):2 with lines ls 4 t 'Quadruped Parallel4', \
"CollectedRunsResultSerialBIPED.gnuplot.txt" using ($1+1):2 with lines ls 5 t 'Biped Serial', \
"CollectedRunsResultParallelBIPED2.gnuplot.txt" using ($1+1):2 with lines ls 6 t 'Biped Parallel2', \
"CollectedRunsResultParallelBIPED3.gnuplot.txt" using ($1+1):2 with lines ls 7 t 'Biped Parallel3', \
"CollectedRunsResultParallelBIPED4.gnuplot.txt" using ($1+1):2 with lines ls 8 t 'Biped Parallel4'




# PDF
set terminal pdf enhanced font 'Calibri,10'
set output "render/collectedRuns_ALL_vector.pdf"
replot



# Live (wxWidgets)
set terminal wxt size out_w,out_h enhanced font 'Verdana,25' persist
replot

