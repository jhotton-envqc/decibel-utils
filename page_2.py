# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 09:56:25 2025

@author: hotju02
"""

import streamlit as st

# =============================================================================
st.markdown("# Calculatrice de décibels")
st.sidebar.markdown("# Calculatrice de décibels")
st.sidebar.write("Équation pour l'addition de décibels:")
st.sidebar.image("static/add.png", use_container_width=True)
st.sidebar.write("Équation pour la soustraction de décibels:")
st.sidebar.image("static/diff.png", use_container_width=True)
# =============================================================================


# Embed HTML content directly
html_code = """
<!doctype html>
<html lang="fr" xmlns:mso="urn:schemas-microsoft-com:office:office" xmlns:msdt="uuid:C2F41010-65B3-11d1-A29F-00AA00C14882">
	<head>
        <!-- <link rel="stylesheet" href="style.css"> -->
        <title></title>
    
<!--[if gte mso 9]><xml>
<mso:CustomDocumentProperties>
<mso:display_urn_x003a_schemas-microsoft-com_x003a_office_x003a_office_x0023_Editor msdt:dt="string">Hotton, Julien</mso:display_urn_x003a_schemas-microsoft-com_x003a_office_x003a_office_x0023_Editor>
<mso:Order msdt:dt="string">90800.0000000000</mso:Order>
<mso:ComplianceAssetId msdt:dt="string"></mso:ComplianceAssetId>
<mso:SharedWithUsers msdt:dt="string"></mso:SharedWithUsers>
<mso:_ExtendedDescription msdt:dt="string"></mso:_ExtendedDescription>
<mso:display_urn_x003a_schemas-microsoft-com_x003a_office_x003a_office_x0023_Author msdt:dt="string">Hotton, Julien</mso:display_urn_x003a_schemas-microsoft-com_x003a_office_x003a_office_x0023_Author>
<mso:TriggerFlowInfo msdt:dt="string"></mso:TriggerFlowInfo>
<mso:ContentTypeId msdt:dt="string">0x010100C1225DEEE7C1AA438BC05E5506038E1E</mso:ContentTypeId>
<mso:_SourceUrl msdt:dt="string"></mso:_SourceUrl>
<mso:_SharedFileIndex msdt:dt="string"></mso:_SharedFileIndex>
</mso:CustomDocumentProperties>
</xml><![endif]-->
</head> 
    <body style="background-color: #202124;">
        <!-- Addition -->
        <form>
            <table>
            <tr>
                <td colspan="2" style="font-weight: bold;
                font-size: 18px; color:#FFF;" align="center">-Addition-</td>
            </tr>
            <tr>    
                <td colspan="2">  
            <label style="font-weight: bold;
            font-size: 16px;
            color: #FFF;
            padding-left: 10px;
            line-height: 1.5;" for="niv1">Niveau 1:</label><br />
            <input style="border-radius: 15px;
            border: 2px solid #3c4043;
            font-size: 18px;
            line-height: 1.5;
            outline: none;
            font-weight: bold;
            color: #FFFFFF;
            padding: 5px 10px 5px 10px;
            background-color: #303134;" type="number" id="niv1" value="40">
        <br /><br />
        </td>  
          
        </tr>
           
            <tr>
                <td colspan="2"> 
            <label style="font-weight: bold;
            font-size: 16px;
            color: #FFF;
            padding-left: 10px;
            line-height: 1.5;"  for="niv2">Niveau 2:</label><br />
            <input style="border-radius: 15px;
            border: 2px solid #3c4043;
            font-size: 18px;
            line-height: 1.5;
            outline: none;
            font-weight: bold;
            color: #FFFFFF;
            padding: 5px 10px 5px 10px;
            background-color: #303134;" type="number" id="niv2" value="40">
            <br /><br />
        </td>      
    </tr>
    <tr>
        <td>
           <label style="font-weight: bold;
            font-size: 16px;
            color: #FFF;
            padding-left: 10px;
            line-height: 1.5;" for="total">Total:</label><br />
           <!--  <input type="number" id="total" value="43"> -->
           <output style="border-radius: 15px;
           border: 2px solid #3c4043;
           font-size: 18px;
           line-height: 1.5;
           outline: none;
           font-weight: bold;
           color: #FFFFFF;
           padding: 5px 80px 5px 10px;
           background-color: #303134;" name="total" id="total">43.01</output>
           <br /><br />
        </td>
        <td>
           <label style="font-weight: bold;
           font-size: 16px;
           color: #FFF;
           padding-left: 10px;
           line-height: 1.5;" for="totalR">Total(arrondi):</label><br />
          <!--  <input type="number" id="totalR" value="43"> -->
          <output style="border-radius: 15px;
          border: 2px solid #3c4043;
          font-size: 18px;
          line-height: 1.5;
          outline: none;
          font-weight: bold;
          color: #FFFFFF;
          padding: 5px 80px 5px 10px;
          background-color: #303134;" name="totalR" id="totalR">43</output>
                     <br /><br />
          </td>
    </tr>
         </table>
        </form>
        <br /><br />
        <!-- Soustraction -->
        <form>
            <table>
            <tr>
                <td colspan="2" style="font-weight: bold;
                font-size: 18px; color:#FFF;" align="center">-Soustraction-</td>
            </tr>
            <tr>    
                <td colspan="2">  
            <label style="font-weight: bold;
            font-size: 16px;
            color: #FFF;
            padding-left: 10px;
            line-height: 1.5;" for="niv1b">Niveau 1:</label><br />
            <input style="border-radius: 15px;
            border: 2px solid #3c4043;
            font-size: 18px;
            line-height: 1.5;
            outline: none;
            font-weight: bold;
            color: #FFFFFF;
            padding: 5px 10px 5px 10px;
            background-color: #303134;" type="number" id="niv1b" value="50">
        <br /><br />
        </td>  
          
        </tr>
           
            <tr>
                <td colspan="2"> 
            <label style="font-weight: bold;
            font-size: 16px;
            color: #FFF;
            padding-left: 10px;
            line-height: 1.5;"  for="niv2b">Niveau 2:</label><br />
            <input style="border-radius: 15px;
            border: 2px solid #3c4043;
            font-size: 18px;
            line-height: 1.5;
            outline: none;
            font-weight: bold;
            color: #FFFFFF;
            padding: 5px 10px 5px 10px;
            background-color: #303134;" type="number" id="niv2b" value="40">
            <br /><br />
        </td>      
    </tr>
    <tr>
        <td>
           <label style="font-weight: bold;
            font-size: 16px;
            color: #FFF;
            padding-left: 10px;
            line-height: 1.5;" for="totalb">Total:</label><br />
           <!--  <input type="number" id="total" value="43"> -->
           <output style="border-radius: 15px;
           border: 2px solid #3c4043;
           font-size: 18px;
           line-height: 1.5;
           outline: none;
           font-weight: bold;
           color: #FFFFFF;
           padding: 5px 80px 5px 10px;
           background-color: #303134;" name="totalb" id="totalb">49.54</output>
           <br /><br />
        </td>
        <td>
           <label style="font-weight: bold;
           font-size: 16px;
           color: #FFF;
           padding-left: 10px;
           line-height: 1.5;" for="totalRb">Total(arrondi):</label><br />
          <!--  <input type="number" id="totalR" value="43"> -->
          <output style="border-radius: 15px;
          border: 2px solid #3c4043;
          font-size: 18px;
          line-height: 1.5;
          outline: none;
          font-weight: bold;
          color: #FFFFFF;
          padding: 5px 80px 5px 10px;
          background-color: #303134;" name="totalRb" id="totalRb">50</output>
                     <br /><br />
          </td>
    </tr>
         </table>
        </form>
        <script>
            window.onload = function() {
                // Addition
                let n1 = document.getElementById("niv1");
                let n2 = document.getElementById("niv2");
                let tot = document.getElementById("total")
                let totR = document.getElementById("totalR")
                n1.oninput = function() {
                     tot.value = (10*Math.log10(Math.pow(10,n1.value/10)+ Math.pow(10,n2.value/10))).toFixed(2);
                     totR.value = Math.round(tot.value);
                };
                n2.oninput = function() {
                    tot.value = (10*Math.log10(Math.pow(10,n1.value/10)+ Math.pow(10,n2.value/10))).toFixed(2);
                    totR.value = Math.round(tot.value);
                 }
                // Soustraction    
                let n1b = document.getElementById("niv1b");
                let n2b = document.getElementById("niv2b");
                let totb = document.getElementById("totalb")
                let totRb = document.getElementById("totalRb")
                n1b.oninput = function() {
                     totb.value = (10*Math.log10(Math.pow(10,n1b.value/10)- Math.pow(10,n2b.value/10))).toFixed(2);
                     totRb.value = Math.round(totb.value);
                };
                n2b.oninput = function() {
                    totb.value = (10*Math.log10(Math.pow(10,n1b.value/10)- Math.pow(10,n2b.value/10))).toFixed(2);
                    totRb.value = Math.round(totb.value);
                 } 
            };
        </script>
    </body>
    </html>

"""
st.components.v1.html(html_code, height=600, width=300)








