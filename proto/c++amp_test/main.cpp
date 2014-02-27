#include <iostream> 
#include <amp.h> 
#include <ppl.h>

using namespace std;
using namespace concurrency; 

int main() 
{ 
	int v[11] = {'G', 'd', 'k', 'k', 'n', 31, 'v', 'n', 'q', 'k', 'c'};

	// Serial (CPU)


	// PPL (CPU)
	int pplRes[11];
	parallel_for(0, 11, [&](int n) {
		pplRes[n]=v[n]+1;
      });
	for(unsigned int i = 0; i < 11; i++) 
		std::cout << static_cast<char>(pplRes[i]); 

	// C++AMP (GPU)
	array_view<int> av(11, v); 
	parallel_for_each(av.extent, [=](index<1> idx) restrict(amp) 
	{ 
		av[idx] += 1; 
	});


	// Print C++AMP
	for(unsigned int i = 0; i < 11; i++) 
		std::cout << static_cast<char>(av[i]); 


	return 0;
}