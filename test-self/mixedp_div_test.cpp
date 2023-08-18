//Copyright (c) 2022-2023, RIKEN Center for Computational Science. All rights reserved.
//SPDX-License-Identifier: BSD-3-Clause
//This program is free software: you can redistribute it and/or modify it under
//the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "tmblas.hpp"
#include "test_util.hpp"

template<typename Th, typename Tl>
void check_div(std::string &st_pi, std::string &st_e, std::string st_log2)
{
  Th oa = tmblas::fromstring<Th>(st_pi);
  Th ob = tmblas::fromstring<Th>(st_e);
  Th of = tmblas::fromstring<Th>(st_log2);
  Th ox(0);
  
  Tl qa = tmblas::type_conv<Tl, Th>(oa); // downcast
  Tl qb = tmblas::type_conv<Tl, Th>(ob);
  Tl qf = tmblas::type_conv<Tl, Th>(of);
  Tl qd(0);

  std::complex<Tl> qac(qa, qb);
  std::complex<Tl> qbc(qb, qf);
  std::complex<Tl> qfc(qf, qa);
  std::complex<Tl> qcc(0), qdc(0), qdr(0);


  std::complex<Th> oac(oa, ob);
  std::complex<Th> obc(ob, of);
  std::complex<Th> ofc(of, oa);
  std::complex<Th> occ(0), odc(0), oxc(0), oxr(0);
  
  std::complex<Th> oar(oa, Th(0));
  std::complex<Th> obr(ob, Th(0));

  std::complex<Tl> qar(qa, Tl(0));
  std::complex<Tl> qbr(qb, Tl(0));

  
  printf("%s %d :: real\n", __FILE__, __LINE__);
  
  tmblas::mixedp_div<Tl, Tl, Tl>(qd, qa, qb);
  printf("QQQ %s\n", tmblas::tostring(qd).c_str());    
  // OQQ
  tmblas::mixedp_div<Th, Tl, Tl>(ox, qa, qb);
  printf("OQQ %s\n", tmblas::tostring(ox).c_str());    
  // OQO
  tmblas::mixedp_div<Th, Tl, Th>(ox, qa, ob);
  printf("OQO %s\n", tmblas::tostring(ox).c_str());  
  // OOQ
  tmblas::mixedp_div<Th, Th, Tl>(ox, oa, qb);
  printf("OOQ %s\n", tmblas::tostring(ox).c_str());  
  
  printf("%s %d :: complex\n", __FILE__, __LINE__);

 // div  
//OOO
//  tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, std::complex<Th> >(odc, oac, obc);
//  printf("OOO %s\n",
//	 tmblas::tostring(odc).c_str());
//
//QQQ
 tmblas::mixedp_div<std::complex<Tl>, std::complex<Tl>, std::complex<Tl> >(qdc, qac, qbc);  
 printf("QQQ %s\n",  tmblas::tostring(qdc).c_str());

 //OQQ
  tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, std::complex<Tl> >(oxc, qac, qbc);  
  printf("OQQ %s\n", tmblas::tostring(oxc).c_str());
//OQO
  tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, std::complex<Th> >(oxc, qac, obc);  
  printf("OQO %s\n", tmblas::tostring(oxc).c_str());
//OOQ
  tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, std::complex<Tl> >(oxc, oac, qbc);  
  printf("OOQ %s\n", tmblas::tostring(oxc).c_str());

  printf("%s %d :: complex - real\n", __FILE__, __LINE__);
//QQQr
  tmblas::mixedp_div<std::complex<Tl>, std::complex<Tl>, Tl>(qdr, qac, qb);
  printf("QQQr %s\n", tmblas::tostring(qdr).c_str());
tmblas::mixedp_div<std::complex<Tl>, std::complex<Tl>, std::complex<Tl> >(qdc, qac, qbr);    
  printf("QQQq %s\n", tmblas::tostring(qdc).c_str());

//QQrQ
tmblas::mixedp_div<std::complex<Tl>, Tl, std::complex<Tl> >(qdr, qa, qbc);
   printf("QQrQ %s\n", tmblas::tostring(qdr).c_str());
tmblas::mixedp_div<std::complex<Tl>, std::complex<Tl>, std::complex<Tl> >(qdc, qar, qbc);    
  printf("QQqQ %s\n", tmblas::tostring(qdc).c_str());
  
//OQQr
tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, Tl>(oxc, qac, qb);
 printf("OQQr %s\n", tmblas::tostring(oxc).c_str());
 tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, std::complex<Tl> >(oxr, qac, qbr);
 printf("OQQq %s\n", tmblas::tostring(oxr).c_str());

//OQrQ
  tmblas::mixedp_div<std::complex<Th>, Tl, std::complex<Tl> >(oxc, qa, qbc);
  printf("OQrQ %s\n", tmblas::tostring(oxc).c_str());
    tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, std::complex<Tl> >(oxr, qar, qbc);
    printf("OQqQ %s\n", tmblas::tostring(oxr).c_str());

//OQOr
  tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, Th>(oxc, qac, ob);
  printf("OQOr %s\n", tmblas::tostring(oxc).c_str());
    tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, std::complex<Th> >(oxr, qac, obr);
    printf("OQOl %s\n", tmblas::tostring(oxr).c_str());

  //OQrO
  tmblas::mixedp_div<std::complex<Th>, Tl, std::complex<Th> >(oxc, qa, obc);
  printf("OQrO %s\n", tmblas::tostring(oxc).c_str());  
  tmblas::mixedp_div<std::complex<Th>, std::complex<Tl>, std::complex<Th> >(oxr, qar, obc);
  printf("OQlO %s\n", tmblas::tostring(oxr).c_str());  

//OOQr
  tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, Tl>(oxc, oac, qb);
    printf("OOQr %s\n", tmblas::tostring(oxc).c_str());  
    tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, std::complex<Tl> >(oxr, oac, qbr);
    printf("OOQl %s\n", tmblas::tostring(oxr).c_str());  

//OOrQ
  tmblas::mixedp_div<std::complex<Th>, Th, std::complex<Tl> >(oxc, oa, qbc);
  printf("OOrQ %s\n", tmblas::tostring(oxc).c_str());  
    tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, std::complex<Tl> >(oxr, oar, qbc);
    printf("OOlQ %s\n", tmblas::tostring(oxr).c_str());  

//OOOr
  tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, Th>(odc, oac, ob);
  printf("OOOr %s\n",
	 tmblas::tostring(odc).c_str());
  tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, std::complex<Th> >(odc, oac, obr);
  printf("OOOl %s\n",
	 tmblas::tostring(odc).c_str());
  // OOrO    
  tmblas::mixedp_div<std::complex<Th>, Th, std::complex<Th> >(odc, oa, obc);
  printf("OOrO %s\n",
	 tmblas::tostring(odc).c_str());
  tmblas::mixedp_div<std::complex<Th>, std::complex<Th>, std::complex<Th> >(odc, oar, obc);
  printf("OOlO %s\n",
	 tmblas::tostring(odc).c_str());
//  
}

int main(int argc, char **argv)
{
  // mathematical constants from www.wolframalpha.com
  std::string st_pi(  "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282");
  std::string st_e(   "2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427466391932003");
  std::string st_log2("0.693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418687542001481020570");

  octuple oa = tmblas::fromstring<octuple>(st_pi);
  octuple ob = tmblas::fromstring<octuple>(st_e);
  octuple of = tmblas::fromstring<octuple>(st_log2);
  octuple oc(0), od(0), ox(0);


  printf("%s %d :: real\n", __FILE__, __LINE__);

  tmblas::mixedp_div<octuple, octuple, octuple>(od, oa, ob);
  printf("OOO %s\n", tmblas::tostring(od).c_str());    
  // QQQ

  // complex
  std::complex<octuple> oac(oa, ob);
  std::complex<octuple> obc(ob, of);
  std::complex<octuple> ofc(of, oa);
  std::complex<octuple> occ(0), odc(0), oxc(0), oxr(0);

  printf("%s %d :: complex\n", __FILE__, __LINE__);

 // div  
//OOO
  tmblas::mixedp_div<std::complex<octuple>, std::complex<octuple>, std::complex<octuple> >(odc, oac, obc);
  printf("OOO %s\n",
	 tmblas::tostring(odc).c_str());
  printf("%s %d: octuple-quaduple\n", __FILE__, __LINE__);
  check_div<octuple, quadruple>(st_pi, st_e, st_log2);
  printf("%s %d: quaduple-double\n", __FILE__, __LINE__);
  check_div<quadruple, double>(st_pi, st_e, st_log2);
  printf("%s %d: double-float\n", __FILE__, __LINE__);
  check_div<double, float>(st_pi, st_e, st_log2);
  
  return 0;

}
