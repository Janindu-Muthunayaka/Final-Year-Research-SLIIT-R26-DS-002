import os

# Your specific file paths
input_json_path = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model\Tree\Sinhala_Tree.json"
output_html_path = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model\Tree\Sinhala_Tree_Viewer.html"

# HTML Template with embedded CSS and Lazy-Loading JavaScript
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sinhala Dictionary Tree Viewer</title>
    <style>
        body { 
            font-family: Consolas, "Courier New", monospace; 
            background-color: #1e1e1e; 
            color: #d4d4d4; 
            padding: 20px; 
            line-height: 1.5;
        }
        .node { 
            margin-left: 20px; 
            list-style-type: none; 
            border-left: 1px dashed #444; 
            padding-left: 10px; 
        }
        .key { 
            color: #9cdcfe; 
            font-weight: bold; 
            cursor: pointer; 
            user-select: none; 
        }
        .key:hover { text-decoration: underline; background-color: #2a2d2e; }
        .value { color: #b5cea8; }
        .string-value { color: #ce9178; }
        .meta-box { 
            background: #252526; 
            border: 1px solid #333; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px; 
            font-family: -apple-system, sans-serif;
        }
        .collapsed::before { content: '▶ '; font-size: 0.8em; color: #888; }
        .expanded::before { content: '▼ '; font-size: 0.8em; color: #888; }
        ul { padding-left: 0; margin-top: 5px; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h2 style="font-family: sans-serif;">Sinhala Character Tree Viewer</h2>
    <div id="meta-container" class="meta-box">Processing massive dataset...</div>
    <div id="tree-container"></div>

    <script>
        // The 28MB JSON string is injected directly here by Python
        const treeData = __JSON_DATA__;

        function renderMeta(meta) {
            if(!meta) return;
            const container = document.getElementById('meta-container');
            container.innerHTML = `<strong>Metadata:</strong><br>
                                   Total Words: ${(meta.total_words || 0).toLocaleString()}<br>
                                   Unique Words: ${(meta.unique_words || 0).toLocaleString()}`;
        }

        function createNode(key, value) {
            const li = document.createElement('li');
            li.className = 'node';

            const span = document.createElement('span');
            span.className = 'key';
            span.textContent = key + ': ';

            if (typeof value === 'object' && value !== null) {
                span.classList.add('collapsed');
                
                // Lazy-loading: Children are only generated when clicked
                span.onclick = function(e) {
                    e.stopPropagation();
                    if (span.classList.contains('collapsed')) {
                        span.classList.replace('collapsed', 'expanded');
                        let ul = li.querySelector('ul');
                        
                        if (!ul) {
                            ul = document.createElement('ul');
                            for (let k in value) {
                                ul.appendChild(createNode(k, value[k]));
                            }
                            li.appendChild(ul);
                        } else {
                            ul.style.display = 'block';
                        }
                    } else {
                        span.classList.replace('expanded', 'collapsed');
                        const ul = li.querySelector('ul');
                        if (ul) ul.style.display = 'none';
                    }
                };
                li.appendChild(span);
                
                const preview = document.createElement('span');
                preview.className = 'value';
                preview.style.color = '#808080';
                preview.style.fontSize = '0.9em';
                preview.textContent = '{ ' + Object.keys(value).length + ' branches }';
                li.appendChild(preview);

            } else {
                li.appendChild(span);
                const valSpan = document.createElement('span');
                valSpan.className = typeof value === 'string' ? 'string-value' : 'value';
                valSpan.textContent = typeof value === 'string' ? `"${value}"` : value;
                li.appendChild(valSpan);
            }

            return li;
        }

        document.addEventListener('DOMContentLoaded', () => {
            if(treeData.meta) {
                renderMeta(treeData.meta);
            }
            
            const rootUl = document.createElement('ul');
            if(treeData.tree) {
                const rootNode = createNode('tree', treeData.tree);
                rootUl.appendChild(rootNode);
                
                // Auto-expand the very first root 'tree' node
                rootNode.querySelector('.key').click();
            } else {
                for (let k in treeData) {
                    if(k !== 'meta') rootUl.appendChild(createNode(k, treeData[k]));
                }
            }
            document.getElementById('tree-container').appendChild(rootUl);
        });
    </script>
</body>
</html>
"""

def main():
    print(f"Reading dataset from {input_json_path}...")
    try:
        # Read the file raw. This avoids the memory overhead of parsing 
        # and re-dumping the 28MB dictionary in Python.
        with open(input_json_path, 'r', encoding='utf-8') as f:
            json_string = f.read()
        
        print("Injecting data and generating HTML viewer...")
        # Inject the raw string into the JavaScript variable
        html_content = html_template.replace("__JSON_DATA__", json_string)
        
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"\nSuccess! ")
        print(f"Double-click the following file to explore your tree:")
        print(f"-> {output_html_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {input_json_path}")
        print("Please check if the path is correct and the file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()